# ai-mux

Strategically alternate between various Vercel AI SDK language models. Good for round-robin cycling API keys, controlled switching between models, and more.

```bash
npm install ai-mux ai
```

### Quick start

```ts
import { generateText } from 'ai';
import { muxModels, roundRobinStrategy } from 'ai-mux';
import { createOpenAI } from '@ai-sdk/openai';
import { createAnthropic } from '@ai-sdk/anthropic';

// muxed automatically cycles between the two API Keys
let muxed = muxApiKeysForModel({
  keys: [API_KEY_1, API_KEY_2],
  createModel: (apiKey) => createOpenAI({ apiKey })('gpt-5.1'),
});

// muxed automatically cycles between the two providers
let muxed = muxModels({
  models: [createOpenAI({ apiKey: process.env.OPENAI_API_KEY })('gpt-5.1'),
           createAnthropic({ apiKey: process.env.ANTHROPIC_API_KEY })('claude-4-5-sonnet')],
  strategy: roundRobinStrategy(),
});

// Use the muxed output as a drop-in replacement for your model.
const { text } = await generateText({ model: muxed, prompt: 'hello there' });
```

### Selection Strategy

There are two prepackaged selection strategies:

- `roundRobinStrategy()` (**Default**): Starts the first request on a random model. Continues to the next model on the next request made in order, cycling back to the first one at the end.
- `randomStrategy()`: Simply chooses a random model with equal probability every single time.

```ts
import { muxModels, type ModelSelectionStrategy } from 'ai-mux';

// Create your own strategy based on the model list, attempt #, and last index selected
const preferPrimary: ModelSelectionStrategy = ({ models, attempt, lastIndex }) => 0 // Index of selected model

const model = muxModels({ models: [MODEL_1, MODEL_2, MODEL_3], strategy: preferPrimary });
```

### Observability

- The `onSelect` callback on both functions provide `{ index, name?, model }` on every call.
- Provider Metadata is attached to generated output. Access it with `providerMetadata["ai-mux"]` (`{ selectedIndex, selectedName?, provider, modelId }`).
