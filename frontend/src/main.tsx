import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { finishSignIn, hub } from "./account";
import App from "./App";
import HoverTip from "./components/HoverTip";
import { LangProvider } from "./i18n";
import "@fontsource/noto-sans/400.css";
import "@fontsource/noto-sans/500.css";
import "@fontsource/noto-sans/600.css";
import "@fontsource/noto-sans/700.css";
import "@fontsource/source-code-pro/400.css";
import "@fontsource/source-code-pro/600.css";
import "./index.css";

// a sign-in link came back with #lk_grant=…: restore the route before the app reads it, then sign in
// otherwise refresh the token once, so a plan an admin just changed shows up on this load
if (!finishSignIn() && hub?.session()) void hub.reload();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <LangProvider>
      <App />
      <HoverTip />
    </LangProvider>
  </StrictMode>,
);
