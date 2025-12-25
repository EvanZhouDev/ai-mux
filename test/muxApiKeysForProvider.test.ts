import { describe, expect, test } from "vitest"
import { muxApiKeysForProvider } from "../src/index.js"

type FakeProviderMetadata = Record<
	string,
	{ selectedIndex?: number; [key: string]: unknown }
>

type FakeModelResult = {
	content: Array<{ type: "text"; text: string }>
	finishReason: "stop"
	usage: { inputTokens: number; outputTokens: number }
	warnings: []
	providerMetadata: FakeProviderMetadata
}

type FakeLanguageModel = {
	specificationVersion: "v3"
	provider: string
	modelId: string
	supportedUrls: Record<string, never>
	doGenerate: (options: unknown) => Promise<FakeModelResult>
	doStream: (options: unknown) => Promise<{ stream: ReadableStream<unknown> }>
}

const createFakeLanguageModel = ({
	apiKey,
	modelId,
	failWith,
}: {
	apiKey: string
	modelId: string
	failWith?: Error | ((options: { apiKey: string; modelId: string }) => Error)
}): FakeLanguageModel => {
	const resolveFailure = () =>
		typeof failWith === "function" ? failWith({ apiKey, modelId }) : failWith

	return {
		specificationVersion: "v3" as const,
		provider: `fake:${apiKey}`,
		modelId,
		supportedUrls: {},
		async doGenerate(_options: unknown) {
			if (failWith !== undefined) {
				throw resolveFailure()
			}

			return {
				content: [{ type: "text", text: `${apiKey}:${modelId}` }],
				finishReason: "stop",
				usage: { inputTokens: 0, outputTokens: 0 },
				warnings: [],
				providerMetadata: {},
			}
		},
		async doStream(_options: unknown) {
			if (failWith !== undefined) {
				throw resolveFailure()
			}

			const stream = new ReadableStream({
				start(controller) {
					controller.enqueue({
						type: "finish",
						usage: { inputTokens: 0, outputTokens: 0 },
						finishReason: "stop",
						providerMetadata: {},
					})
					controller.close()
				},
			})

			return { stream }
		},
	}
}

describe("muxApiKeysForProvider", () => {
	test("shares selection across models", async () => {
		const selections: number[] = []

		const provider = muxApiKeysForProvider({
			keys: ["k1", "k2"],
			createProvider: (apiKey) => (modelId: string) =>
				createFakeLanguageModel({ apiKey, modelId }),
			strategy: ({ models, attempt }) => attempt % models.length,
			onSelect: ({ index }) => {
				selections.push(index)
			},
		})

		const modelA = provider("a")
		const modelB = provider("b")

		const resA = await modelA.doGenerate({})
		const resB = await modelB.doGenerate({})

		expect(resA.providerMetadata["ai-mux"]?.selectedIndex).toBe(0)
		expect(resB.providerMetadata["ai-mux"]?.selectedIndex).toBe(1)
		expect(selections.slice(0, 2)).toEqual([0, 1])
	})

	test("retries next key on error by default", async () => {
		const rateLimitError = Object.assign(new Error("rate limit exceeded"), {
			statusCode: 429,
		})

		const provider = muxApiKeysForProvider({
			keys: ["bad", "good"],
			createProvider: (apiKey) => (modelId: string) =>
				createFakeLanguageModel({
					apiKey,
					modelId,
					failWith: apiKey === "bad" ? rateLimitError : undefined,
				}),
			strategy: () => 0,
		})

		const model = provider("a")
		const res = await model.doGenerate({})

		expect(res.providerMetadata["ai-mux"]?.selectedIndex).toBe(1)
		expect(res.content).toEqual([{ type: "text", text: "good:a" }])
	})

	test("retries on listed 400 status", async () => {
		const badRequestError = Object.assign(new Error("bad request"), {
			statusCode: 400,
		})

		const provider = muxApiKeysForProvider({
			keys: ["bad", "good"],
			createProvider: (apiKey) => (modelId: string) =>
				createFakeLanguageModel({
					apiKey,
					modelId,
					failWith: apiKey === "bad" ? badRequestError : undefined,
				}),
			strategy: () => 0,
		})

		const model = provider("a")

		const res = await model.doGenerate({})

		expect(res.providerMetadata["ai-mux"]?.selectedIndex).toBe(1)
		expect(res.content).toEqual([{ type: "text", text: "good:a" }])
	})

	test("skips retry for non-listed status codes", async () => {
		const unauthorizedError = Object.assign(new Error("unauthorized"), {
			statusCode: 401,
		})

		const provider = muxApiKeysForProvider({
			keys: ["bad", "good"],
			createProvider: (apiKey) => (modelId: string) =>
				createFakeLanguageModel({
					apiKey,
					modelId,
					failWith: apiKey === "bad" ? unauthorizedError : undefined,
				}),
			strategy: () => 0,
		})

		const model = provider("a")

		await expect(model.doGenerate({})).rejects.toThrow("unauthorized")
	})

	test("can disable retryOnError", async () => {
		const provider = muxApiKeysForProvider({
			keys: ["bad", "good"],
			createProvider: (apiKey) => (modelId: string) =>
				createFakeLanguageModel({
					apiKey,
					modelId,
					failWith: apiKey === "bad" ? new Error(`fail:${apiKey}`) : undefined,
				}),
			strategy: () => 0,
			retryOnError: false,
		})

		const model = provider("a")

		await expect(model.doGenerate({})).rejects.toThrow("fail:bad")
	})
})
