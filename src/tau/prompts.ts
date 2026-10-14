/** Prompts copied verbatim from tau_bench/envs/user.py (LLMUserSimulationEnv). */

export const USER_OPENING = "Hi! How can I help you today?";

export function userSystemPrompt(instruction: string | null): string {
  const instructionDisplay = instruction !== null ? "\n\nInstruction: " + instruction + "\n" : "";
  return `You are a user interacting with an agent.${instructionDisplay}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction.`;
}
