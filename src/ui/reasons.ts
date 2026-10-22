import type { TFunction, TKey } from "../i18n";
import type { EndReason } from "../tau/episode";

const KEYS: Record<EndReason, TKey> = {
  user_stop: "doneUserStop",
  transfer: "doneTransfer",
  max_steps: "doneMaxSteps",
  manual: "doneManual",
  crash: "doneCrash",
  error: "doneError",
  incomplete: "doneIncomplete",
};

/** How an episode ended, in words (falls back to the code for unknown values from old storage). */
export function endReasonText(t: TFunction, reason: string): string {
  const key = KEYS[reason as EndReason];
  return key ? t(key) : reason;
}
