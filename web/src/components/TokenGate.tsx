// Phase 9 (9c): the bearer-token entry.
//
// AUTH MODEL. The bearer token entered here is the ONLY credential the browser
// holds. It is sent as `Authorization: Bearer <token>` on every API call; the
// backend derives ownerId from it (the client NEVER sends an owner/user id). The
// token is held in memory by App state (optionally mirrored to sessionStorage so
// a reload within the tab keeps the session) — it is NEVER baked into the bundle.
// NO provider credential (LITELLM_*) is ever entered or stored here.

import { useState } from "react";

interface TokenGateProps {
  onSubmit: (token: string) => void;
}

export function TokenGate(props: TokenGateProps): React.ReactElement {
  const { onSubmit } = props;
  const [value, setValue] = useState("");

  return (
    <form
      className="token-gate"
      data-testid="token-gate"
      onSubmit={(e) => {
        e.preventDefault();
        const trimmed = value.trim();
        if (trimmed !== "") onSubmit(trimmed);
      }}
    >
      <label className="token-gate__label" htmlFor="token-input">
        Access token
      </label>
      <p className="token-gate__hint">
        Enter your bearer token. It authenticates you to the conversation
        service; your ownership is derived server-side from this token. No
        provider credentials are used or stored in this browser app.
      </p>
      <input
        id="token-input"
        data-testid="token-input"
        type="password"
        autoComplete="off"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="paste bearer token"
      />
      <button type="submit" data-testid="token-submit" disabled={value.trim() === ""}>
        Continue
      </button>
    </form>
  );
}
