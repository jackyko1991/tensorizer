import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
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

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <LangProvider>
      <App />
      <HoverTip />
    </LangProvider>
  </StrictMode>,
);
