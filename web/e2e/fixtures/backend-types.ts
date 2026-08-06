// Phase 9 (9d): NARROW backend type surface for the E2E harness.
//
// WHY THIS FILE. The harness (test-server.ts) boots the REAL backend
// `createServer(deps)` at RUNTIME (dynamic import, resolved against the root
// project's node_modules by tsx). But `web/` is a SEPARATE tsconfig project and
// must NOT type-check the backend `src/**` (different lib/target/strictness — it
// would drag the whole backend, and its exactOptionalPropertyTypes-incompatible
// third-party types, into the web build). So — exactly like the 9c SSE-contract
// mirror (src/lib/sse-contract.ts) — we RE-DECLARE here ONLY the narrow slice of
// the backend surface the harness uses, and cast the dynamic imports to it.
//
// DRIFT is caught by the E2E ITSELF: the harness drives the REAL server, and the
// specs assert the browser renders the scripted app-owned citation fields — a
// mismatch between this mirror and the backend would surface as a failed boot or
// a failed journey. Keep this in lockstep with the backend types it names.

// Mirror of src/core/sse.ts EvidenceStatus / src/core/rag.ts.
export type EvidenceStatus = "supported" | "insufficient";

// Mirror of src/core/citations.ts TrustedCitation (every field APP-OWNED). The
// server drops `ownerId` from the wire; the harness still sets it so the scripted
// result is authorization-consistent with the caller.
export interface TrustedCitation {
  marker: number;
  chunkId: string;
  documentId: string;
  sourceUri: string | null;
  chunkIndex: number;
  ownerId: string | null;
  contentHash: string;
  score: number;
}

// Mirror of src/core/schemas.ts GroundedAnswer.
export interface GroundedAnswer {
  answer: string;
  citations: number[];
  insufficientEvidence: boolean;
}

// The subset of src/core/rag.ts GroundedAnswerResult the server reads
// (answer.answer, resolvedCitations, evidenceStatus). The other fields are
// carried structurally so the scripted value satisfies the real type at runtime.
export interface GroundedAnswerResult {
  answer: GroundedAnswer;
  resolvedCitations: TrustedCitation[];
  droppedCitations: unknown[];
  markerMap: Map<number, unknown>;
  retrieved: unknown[];
  contextBlock: string;
  evidenceStatus: EvidenceStatus;
}

// Mirror of src/core/grounded-turn.ts AnswerTurn args + type.
export interface AnswerTurnArgs {
  query: string;
  ownerId: string;
  signal: AbortSignal;
}
export type AnswerTurn = (args: AnswerTurnArgs) => Promise<GroundedAnswerResult>;

// Mirror of src/core/grounded-turn.ts TurnAbortedError (constructor only).
export interface TurnAbortedErrorCtor {
  new (): Error;
}

// Mirror of the src/core/threads.ts Thread record + the ThreadStore surface the
// harness uses (createThread only). Kept narrow.
export interface Thread {
  id: string;
  ownerId: string;
  title: string | null;
  createdAt: number;
  updatedAt: number;
}
export interface ThreadStore {
  createThread(ownerId: string, opts?: { title?: string }): Thread;
}

// Mirror of the StreamingChat factory return the server's createChat expects.
// The harness never actually produces one (createChat throws), so `unknown` is
// the safe return without importing the backend's BaseMessage types.
export type StreamingChatFactory = () => never;

// The narrow shape of src/core/auth.ts TokenAuthenticator (opaque to the
// harness, which only passes it through to createServer).
export interface TokenAuthenticator {
  readonly size: number;
}

// The narrow ServerDeps the harness constructs. `store`/`auth` are typed as the
// concrete harness values; the model/grounded seams are the faked boundary.
export interface ServerDeps {
  store: ThreadStore;
  auth: TokenAuthenticator;
  createChat: StreamingChatFactory;
  answerTurn?: AnswerTurn;
  log?: (line: string) => void;
}

// An http.Server-like handle (only listen/close are used by the harness).
export interface HttpServerLike {
  listen(port: number, host: string, cb: () => void): unknown;
  close(cb?: () => void): unknown;
}

// The narrow backend module surface the harness dynamic-imports and casts to.
export interface BackendServerModule {
  createServer(deps: ServerDeps): HttpServerLike;
}
export interface BackendMemoryModule {
  // Returns a saver; the harness narrows it to the SqliteSaver-with-db shape.
  createCheckpointer(dbPath?: string): SqliteSaverLike;
}
export interface SqliteSaverLike {
  db: { close(): void };
}
export interface BackendThreadsModule {
  createThreadStore(saver: SqliteSaverLike): ThreadStore;
}
export interface BackendAuthModule {
  createTokenAuthenticator(
    tokenToUserId: Readonly<Record<string, string>>
  ): TokenAuthenticator;
}
export interface BackendGroundedTurnModule {
  TurnAbortedError: TurnAbortedErrorCtor;
}
