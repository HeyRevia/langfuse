import { z } from "zod";
import { ChatMessageRole, LLMAdapter } from "@langfuse/shared";

const ModelParamsSchema = z.object({
  provider: z.string(),
  adapter: z.nativeEnum(LLMAdapter),
  model: z.string(),
  temperature: z.number().optional(),
  max_tokens: z.number().optional(),
  top_p: z.number().optional(),
});
const MessageSchema = z.object({
  role: z.nativeEnum(ChatMessageRole),
  content: z.string(),
  id: z.string().optional(),
});

const LLMFunctionCallSchema = z.object({
  name: z.string(),
  parameters: z.record(z.any()),
});

export const ChatCompletionBodySchema = z.object({
  projectId: z.string(),
  messages: z.array(MessageSchema),
  modelParams: ModelParamsSchema,
  functions: z.array(LLMFunctionCallSchema).optional(),
});

export const validateChatCompletionBody = (input: unknown) => {
  return ChatCompletionBodySchema.parse(input);
};

export type ValidatedChatCompletionBody = z.infer<
  typeof ChatCompletionBodySchema
>;
