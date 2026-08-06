// Phase 9 (9c): the SPA entrypoint. Mounts the React app into #root.
//
// NO provider credential is referenced here or anywhere in the frontend. The app
// only ever talks to the backend conversation service (same-origin / dev proxy).

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App.js";
import "./index.css";

const container = document.getElementById("root");
if (container === null) {
  throw new Error("Root container #root not found");
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>
);
