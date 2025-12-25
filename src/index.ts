import type { JSONValue, LanguageModelV3 } from "@ai-sdk/provider"

type NamedModel = { model: LanguageModelV3; name?: string }
type ProviderMetadata = Record<string, Record<string, JSONValue | undefined>>

class AllMuxModelsFailedError extends Error {
	readonly errors: ReadonlyArray<unknown>

	constructor(errors: ReadonlyArray<unknown>) {
		super("ai-mux: all models failed")
		this.name = "AllMuxModelsFailedError"
		this.errors = errors
	}
}

export type LanguageModelCandidate = LanguageModelV3 | NamedModel
export type ModelSelectionStrategy = (ctx: {
	models: readonly NamedModel[]
	attempt: number
	lastIndex: number | null
}) => number
export type MuxModelsOptions = {
	models: ReadonlyArray<LanguageModelCandidate>
	strategy?: ModelSelectionStrategy | "roundRobin" | "random"
	/**
	 * When enabled, a failed request will be retried against the next model/key
	 * (cyclically) if the error status is one of: 400, 403, 429, 500, 503, 504,
	 * until one succeeds or all have failed.
	 */
	retryOnError?: boolean
	onSelect?: (selection: {
		index: number
		name?: string
		model: LanguageModelV3
	}) => void
}
export type ApiKeyProviderMuxOptions<Provider extends object> = {
	keys: ReadonlyArray<string>
	createProvider: (apiKey: string, meta: { index: number }) => Provider
	strategy?: ModelSelectionStrategy | "roundRobin" | "random"
	/**
	 * Defaults to `true`. When enabled, a failed request will be retried using
	 * the next API key (cyclically) if the error status is one of: 400, 403, 429,
	 * 500, 503, 504, until one succeeds or all have failed.
	 */
	retryOnError?: boolean
	onSelect?: MuxModelsOptions["onSelect"]
}

const isNamedModel = (value: LanguageModelCandidate): value is NamedModel => {
	return (
		typeof value === "object" && value !== null && Object.hasOwn(value, "model")
	)
}

const resolveStrategy = (
	strategy: ModelSelectionStrategy | "roundRobin" | "random" | undefined,
): ModelSelectionStrategy => {
	if (strategy === "random") return randomStrategy()
	if (strategy === undefined || strategy === "roundRobin")
		return roundRobinStrategy()
	return strategy
}

export const roundRobinStrategy = (): ModelSelectionStrategy => {
	let cursor = 0
	let seeded = false
	return ({ models }) => {
		if (!seeded) {
			cursor = Math.floor(Math.random() * models.length)
			seeded = true
		}
		const pick = cursor % models.length
		cursor = (cursor + 1) % models.length
		return pick
	}
}

export const randomStrategy =
	(): ModelSelectionStrategy =>
	({ models }) =>
		Math.floor(Math.random() * models.length)

const keyRetryStatusCodes = new Set([400, 403, 429, 500, 503, 504])

const isRecord = (value: unknown): value is Record<string, unknown> =>
	typeof value === "object" && value !== null

const getNumberProperty = (
	value: Record<string, unknown>,
	key: string,
): number | undefined => {
	const candidate = value[key]
	if (typeof candidate === "number") {
		if (!Number.isFinite(candidate)) return undefined
		return candidate
	}
	if (typeof candidate === "string") {
		const parsed = Number(candidate)
		if (!Number.isFinite(parsed)) return undefined
		return parsed
	}
	return undefined
}

const getArrayProperty = (
	value: Record<string, unknown>,
	key: string,
): ReadonlyArray<unknown> | undefined => {
	const candidate = value[key]
	return Array.isArray(candidate) ? candidate : undefined
}

const collectStatusCodes = (error: unknown) => {
	const statusCodes: number[] = []
	const visited = new WeakSet<object>()

	const visit = (value: unknown) => {
		if (value === null || value === undefined) return
		if (!isRecord(value)) return
		if (visited.has(value)) return
		visited.add(value)

		const statusCode =
			getNumberProperty(value, "statusCode") ??
			getNumberProperty(value, "status")
		if (statusCode !== undefined) {
			statusCodes.push(statusCode)
		}

		const cause = value.cause
		if (cause !== undefined) visit(cause)

		const lastError = value.lastError
		if (lastError !== undefined) visit(lastError)

		const errors = getArrayProperty(value, "errors")
		if (errors !== undefined) {
			for (const entry of errors) visit(entry)
		}

		const data = value.data
		if (data !== undefined) visit(data)

		const nestedError = value.error
		if (nestedError !== undefined) visit(nestedError)
	}

	visit(error)
	return statusCodes
}

const shouldRetryWithNextKey = (error: unknown): boolean => {
	const statusCodes = collectStatusCodes(error)
	return statusCodes.some((code) => keyRetryStatusCodes.has(code))
}

export function muxModels(options: MuxModelsOptions): LanguageModelV3 {
	const rawModels = options.models
	const pick = resolveStrategy(options.strategy)
	const retryOnError = options.retryOnError ?? false
	const onSelect = options.onSelect

	if (rawModels.length === 0) {
		throw new Error("muxModels requires at least one model.")
	}

	const entries: NamedModel[] = rawModels.map((entry) =>
		isNamedModel(entry) ? entry : { model: entry },
	)
	const models = entries.map((entry) => entry.model)

	let attempt = 0
	let lastIndex: number | null = null

	const chooseStartIndex = () => {
		const choice = pick({
			models: entries,
			attempt: attempt++,
			lastIndex,
		})
		const numericChoice = Number.isFinite(choice) ? Math.trunc(choice) : 0
		const safeIndex =
			((numericChoice % models.length) + models.length) % models.length
		return safeIndex
	}

	const orderedIndicesFrom = (startIndex: number): ReadonlyArray<number> => {
		const length = models.length
		return Array.from({ length }, (_, offset) => (startIndex + offset) % length)
	}

	const attemptCall = async <T>(
		startIndex: number,
		call: (selection: NamedModel) => PromiseLike<T>,
	): Promise<{ index: number; selection: NamedModel; result: T }> => {
		const indicesToTry = retryOnError
			? orderedIndicesFrom(startIndex)
			: [startIndex]

		const errors: unknown[] = []

		for (const index of indicesToTry) {
			const selection = entries[index]
			if (selection === undefined) {
				throw new Error(
					`ai-mux: internal error: missing model at index ${index}`,
				)
			}
			try {
				const result = await call(selection)
				lastIndex = index
				onSelect?.({ index, name: selection.name, model: selection.model })
				return { index, selection, result }
			} catch (error) {
				if (!retryOnError || !shouldRetryWithNextKey(error)) {
					throw error
				}
				errors.push(error)
			}
		}

		if (errors.length === 1) {
			throw errors[0]
		}
		throw new AllMuxModelsFailedError(errors)
	}

	const supportedUrls = (async () => {
		const allSupported = await Promise.all(
			entries.map((entry) => entry.model.supportedUrls),
		)
		if (allSupported.length === 0) return {}

		const result: Record<string, RegExp[]> = {}
		const first = allSupported[0]
		if (first === undefined) return result

		for (const [mediaType, patterns] of Object.entries(first)) {
			const patternMap = new Map(patterns.map((p) => [p.toString(), p]))

			for (let i = 1; i < allSupported.length && patternMap.size > 0; i++) {
				const supportedUrls = allSupported[i]
				if (supportedUrls === undefined) continue
				const currentSet = new Set(
					(supportedUrls[mediaType] ?? []).map((p) => p.toString()),
				)

				for (const key of Array.from(patternMap.keys())) {
					if (!currentSet.has(key)) patternMap.delete(key)
				}
			}

			if (patternMap.size > 0) {
				result[mediaType] = Array.from(patternMap.values())
			}
		}

		return result
	})()

	const addMuxMetadata = (
		existing: ProviderMetadata | undefined,
		index: number,
		selection: NamedModel,
	): ProviderMetadata => {
		const muxMeta: Record<string, JSONValue> = {
			selectedIndex: index,
			provider: selection.model.provider,
			modelId: selection.model.modelId,
		}
		if (selection.name !== undefined) muxMeta.selectedName = selection.name
		return { ...(existing ?? {}), "ai-mux": muxMeta }
	}

	return {
		specificationVersion: "v3",
		provider: "ai-mux",
		modelId: "mux",
		supportedUrls,
		async doGenerate(options: Parameters<LanguageModelV3["doGenerate"]>[0]) {
			const startIndex = chooseStartIndex()
			const { index, selection, result } = await attemptCall(startIndex, (s) =>
				s.model.doGenerate(options),
			)

			return {
				...result,
				providerMetadata: addMuxMetadata(
					result.providerMetadata,
					index,
					selection,
				),
			}
		},
		async doStream(options: Parameters<LanguageModelV3["doStream"]>[0]) {
			const startIndex = chooseStartIndex()
			const { index, selection, result } = await attemptCall(startIndex, (s) =>
				s.model.doStream(options),
			)

			const reader = result.stream.getReader()
			const mapped = new ReadableStream({
				async pull(controller) {
					const { done, value } = await reader.read()
					if (done) {
						controller.close()
						return
					}
					if (value.type === "finish") {
						controller.enqueue({
							...value,
							providerMetadata: addMuxMetadata(
								value.providerMetadata,
								index,
								selection,
							),
						})
						return
					}
					controller.enqueue(value)
				},
				cancel(reason) {
					return reader.cancel(reason)
				},
			})

			return {
				...result,
				stream: mapped,
			}
		},
	}
}

const isLanguageModelV3 = (value: unknown): value is LanguageModelV3 => {
	if (!isRecord(value)) return false
	if (value.specificationVersion !== "v3") return false
	if (typeof value.provider !== "string") return false
	if (typeof value.modelId !== "string") return false
	if (typeof value.doGenerate !== "function") return false
	if (typeof value.doStream !== "function") return false
	return "supportedUrls" in value
}

export function muxApiKeysForProvider<Provider extends object>(
	options: ApiKeyProviderMuxOptions<Provider>,
): Provider {
	if (options.keys.length === 0) {
		throw new Error("muxApiKeysForProvider requires at least one API key.")
	}

	const retryOnError = options.retryOnError ?? true
	let sharedAttempt = 0
	let sharedLastIndex: number | null = null

	const basePick = resolveStrategy(options.strategy)
	const pick: ModelSelectionStrategy = ({ models }) =>
		basePick({ models, attempt: sharedAttempt++, lastIndex: sharedLastIndex })

	const onSelect: MuxModelsOptions["onSelect"] = (selection) => {
		sharedLastIndex = selection.index
		options.onSelect?.(selection)
	}

	const providers = options.keys.map((key, index) =>
		options.createProvider(key, { index }),
	)

	const muxLanguageModels = (models: ReadonlyArray<LanguageModelV3>) =>
		muxModels({
			models,
			strategy: pick,
			retryOnError,
			onSelect,
		})

	const baseProvider = providers[0]
	if (baseProvider === undefined) {
		throw new Error("muxApiKeysForProvider requires at least one API key.")
	}
	if (typeof baseProvider !== "function") {
		throw new Error(
			"muxApiKeysForProvider createProvider must return a callable provider.",
		)
	}

	return new Proxy(baseProvider, {
		apply(_target, _thisArg, argArray) {
			const models: LanguageModelV3[] = []
			for (const [index, provider] of providers.entries()) {
				if (typeof provider !== "function") {
					throw new Error(
						`muxApiKeysForProvider provider at index ${index} is not callable.`,
					)
				}

				const model = Reflect.apply(provider, provider, argArray)
				if (!isLanguageModelV3(model)) {
					throw new Error(
						`muxApiKeysForProvider provider at index ${index} did not return a v3 LanguageModel.`,
					)
				}

				models.push(model)
			}

			return muxLanguageModels(models)
		},
		get(target, prop, receiver) {
			const value = Reflect.get(target, prop, receiver)
			if (typeof value !== "function") return value

			return (...args: unknown[]) => {
				const primaryResult = Reflect.apply(value, baseProvider, args)
				if (!isLanguageModelV3(primaryResult)) return primaryResult

				const models: LanguageModelV3[] = [primaryResult]
				for (const [index, provider] of providers.entries()) {
					if (index === 0) continue

					const method = Reflect.get(provider, prop)
					if (typeof method !== "function") {
						throw new Error(
							`muxApiKeysForProvider provider at index ${index} is missing a callable ${String(
								prop,
							)}.`,
						)
					}

					const nextResult = Reflect.apply(method, provider, args)
					if (!isLanguageModelV3(nextResult)) {
						throw new Error(
							`muxApiKeysForProvider provider at index ${index} ${String(
								prop,
							)} did not return a v3 LanguageModel.`,
						)
					}
					models.push(nextResult)
				}

				return muxLanguageModels(models)
			}
		},
	})
}
