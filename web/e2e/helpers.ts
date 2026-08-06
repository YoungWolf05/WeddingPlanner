// Phase 9 (9d): shared spec helpers — small, typed page-driving primitives so
// each spec reads as a journey. They use ONLY the stable data-testids the 9c
// components expose. No `any`.

import { expect, type Page } from "@playwright/test";

// Sign in through the token gate with a bearer token. Asserts we land on the
// conversation view (the sign-out affordance is present).
export async function signInWith(page: Page, token: string): Promise<void> {
  await page.goto("/");
  await expect(page.getByTestId("token-gate")).toBeVisible();
  await page.getByTestId("token-input").fill(token);
  await page.getByTestId("token-submit").click();
  await expect(page.getByTestId("sign-out")).toBeVisible();
}

// Create a new thread and wait for it to appear + become the active conversation
// (the composer input is enabled once a thread is selected).
export async function createThread(page: Page): Promise<void> {
  await page.getByTestId("new-thread").click();
  await expect(page.getByTestId("thread-item").first()).toBeVisible();
  await expect(page.getByTestId("composer-input")).toBeEnabled();
}

// Send a message via the composer (types + submits).
export async function sendMessage(page: Page, message: string): Promise<void> {
  await page.getByTestId("composer-input").fill(message);
  await page.getByTestId("send-button").click();
}
