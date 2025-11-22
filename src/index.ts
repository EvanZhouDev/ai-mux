import type { LanguageModel, JSONValue } from "ai";

type LanguageModelV2 = Extract<LanguageModel, { specificationVersion: "v2" }>;
type NamedModel = { model: LanguageModelV2; name?: string };
type ProviderMetadata = Record<string, Record<string, JSONValue>>;

export type LanguageModelCandidate = LanguageModelV2 | NamedModel;
export type ModelSelectionStrategy = (ctx: {
	models: readonly NamedModel[];
	attempt: number;
	lastIndex: number | null;
}) => number;
export type MuxModelsOptions = {
	models: ReadonlyArray<LanguageModelCandidate>;
	strategy?: ModelSelectionStrategy | "roundRobin" | "random";
	onSelect?: (selection: {
		index: number;
		name?: string;
		model: LanguageModelV2;
	}) => void;
};
export type ApiKeyModelMuxOptions = {
	keys: ReadonlyArray<string>;
	createModel: (apiKey: string, meta: { index: number }) => LanguageModelV2;
	strategy?: ModelSelectionStrategy | "roundRobin" | "random";
};

const isNamedModel = (value: LanguageModelCandidate): value is NamedModel => {
	return (
		typeof value === "object" &&
		value !== null &&
		Object.prototype.hasOwnProperty.call(value, "model")
	);
};

export const roundRobinStrategy = (): ModelSelectionStrategy => {
	let cursor = 0;
	let seeded = false;
	return ({ models }) => {
		if (!seeded) {
			cursor = Math.floor(Math.random() * models.length);
			seeded = true;
		}
		const pick = cursor % models.length;
		cursor = (cursor + 1) % models.length;
		return pick;
	};
};

export const randomStrategy =
	(): ModelSelectionStrategy =>
	({ models }) =>
		Math.floor(Math.random() * models.length);

export function muxModels(options: MuxModelsOptions): LanguageModelV2 {
	const rawModels = options.models;
	const rawStrategy = options.strategy;
	const onSelect = options.onSelect;

	if (rawModels.length === 0) {
		throw new Error("muxModels requires at least one model.");
	}

	const entries: NamedModel[] = rawModels.map((entry) =>
		isNamedModel(entry) ? entry : { model: entry }
	);
	const models = entries.map((entry) => entry.model);

	let pick: ModelSelectionStrategy;
	if (rawStrategy === "random") pick = randomStrategy();
	else if (rawStrategy === "roundRobin" || rawStrategy === undefined)
		pick = roundRobinStrategy();
	else pick = rawStrategy;

	let attempt = 0;
	let lastIndex: number | null = null;

	const choose = () => {
		const choice = pick({ models: entries, attempt: attempt++, lastIndex });
		const numericChoice = Number.isFinite(choice) ? Math.trunc(choice) : 0;
		const safeIndex =
			((numericChoice % models.length) + models.length) % models.length;
		lastIndex = safeIndex;
		const selection = entries[safeIndex];
		onSelect?.({ index: safeIndex, name: selection.name, model: selection.model });
		return { index: safeIndex, selection };
	};

	const supportedUrls = (async () => {
		const allSupported = await Promise.all(
			entries.map((entry) => entry.model.supportedUrls)
		);
		if (allSupported.length === 0) return {};

		const result: Record<string, RegExp[]> = {};
		const first = allSupported[0];

		for (const [mediaType, patterns] of Object.entries(first)) {
			const patternMap = new Map(patterns.map((p) => [p.toString(), p]));

			for (let i = 1; i < allSupported.length && patternMap.size > 0; i++) {
				const currentSet = new Set(
					(allSupported[i][mediaType] ?? []).map((p) => p.toString())
				);

				for (const key of Array.from(patternMap.keys())) {
					if (!currentSet.has(key)) patternMap.delete(key);
				}
			}

			if (patternMap.size > 0) {
				result[mediaType] = Array.from(patternMap.values());
			}
		}

		return result;
	})();

	const addMuxMetadata = (
		existing: ProviderMetadata | undefined,
		index: number,
		selection: NamedModel
	): ProviderMetadata => {
		const muxMeta: Record<string, JSONValue> = {
			selectedIndex: index,
			provider: selection.model.provider,
			modelId: selection.model.modelId,
		};
		if (selection.name !== undefined) muxMeta.selectedName = selection.name;
		return { ...(existing ?? {}), "ai-mux": muxMeta };
	};

	return {
		specificationVersion: "v2",
		provider: "ai-mux",
		modelId: "mux",
		supportedUrls,
		doGenerate(options) {
			const choice = choose();
			return choice.selection.model.doGenerate(options).then((result) => ({
				...result,
				providerMetadata: addMuxMetadata(
					result.providerMetadata as ProviderMetadata | undefined,
					choice.index,
					choice.selection
				),
			}));
		},
		doStream(options) {
			const choice = choose();
			return choice.selection.model.doStream(options).then((result) => {
				const reader = result.stream.getReader();
				const mapped = new ReadableStream({
					async pull(controller) {
						const { done, value } = await reader.read();
						if (done) {
							controller.close();
							return;
						}
						if (
							value &&
							typeof value === "object" &&
							"type" in value &&
							value.type === "finish"
						) {
							const finish = value as {
								type: string;
								providerMetadata?: ProviderMetadata;
							};
							controller.enqueue({
								...value,
								providerMetadata: addMuxMetadata(
									finish.providerMetadata,
									choice.index,
									choice.selection
								),
							});
							return;
						}
						controller.enqueue(value);
					},
					cancel(reason) {
						return reader.cancel(reason);
					},
				});

				return {
					...result,
					stream: mapped,
				};
			});
		},
	};
}

export function muxApiKeysForModel(
	options: ApiKeyModelMuxOptions
): LanguageModelV2 {
	if (options.keys.length === 0) {
		throw new Error("muxApiKeysForModel requires at least one API key.");
	}

	const models = options.keys.map((key, index) =>
		options.createModel(key, { index })
	);
	return muxModels({ models, strategy: options.strategy });
}
