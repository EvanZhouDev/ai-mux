# ai-mux

Strategically alternate between various Vercel AI SDK language models

```bash
npm install ai-mux ai
```

### Quick start

```ts
import { generateText } from 'ai';
import { muxModels, roundRobinStrategy } from 'ai-mux';
import { createOpenAI } from '@ai-sdk/openai';
import { createAnthropic } from '@ai-sdk/anthropic';

const muxed = muxModels({
  models: [createOpenAI({ apiKey: process.env.OPENAI_API_KEY })('gpt-5.1'),
           createAnthropic({ apiKey: process.env.ANTHROPIC_API_KEY })('claude-4-5-sonnet')],
  strategy: roundRobinStrategy(),
});

const { text } = await generateText({ model: muxed, prompt: 'hello there' });
```

### API key rotation

```ts
import { muxApiKeysForModel } from 'ai-mux';
import { createOpenAI } from '@ai-sdk/openai';

const muxed = muxApiKeysForModel({
  keys: [process.env.OPENAI_KEY_1!, process.env.OPENAI_KEY_2!],
  createModel: (apiKey) => createOpenAI({ apiKey })('gpt-5.1'),
});
```

### Custom strategy

```ts
import { muxModels, type ModelSelectionStrategy } from 'ai-mux';

const preferPrimary: ModelSelectionStrategy = ({ models, lastIndex }) =>
  lastIndex === null ? 0 : 1 + Math.floor(Math.random() * Math.max(1, models.length - 1));

const model = muxModels({ models: [a, b, c], strategy: preferPrimary });
```

### Observability

- Pass `onSelect` to get `{ index, name?, model }` every call.
- Non-streaming calls add `providerMetadata["ai-mux"]` with `selectedIndex`, `selectedName?`, `provider`, and `modelId`.
