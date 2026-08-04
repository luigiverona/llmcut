import assert from "node:assert/strict";
import test from "node:test";

import { timeoutMilliseconds } from "../src/timeout.ts";

test("converts configured seconds to milliseconds", () => {
  assert.equal(timeoutMilliseconds(), 30_000);
});
