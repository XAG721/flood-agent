import React from "react";
import ReactDOM from "react-dom/client";
import "cesium/Build/Cesium/Widgets/widgets.css";
import App from "./App";
import "./styles.css";

type BoundaryProps = { children: React.ReactNode };
type BoundaryState = { errorMessage: string | null };

class RootErrorBoundary extends React.Component<BoundaryProps, BoundaryState> {
  state: BoundaryState = { errorMessage: null };

  static getDerivedStateFromError(error: unknown): BoundaryState {
    return {
      errorMessage: error instanceof Error ? error.message : String(error)
    };
  }

  componentDidCatch(error: unknown) {
    console.error("Root render failed:", error);
  }

  render() {
    if (this.state.errorMessage) {
      return (
        <div className="overlay">
          <div className="overlay-card">
            <h2>前端渲染失败</h2>
            <p>{this.state.errorMessage}</p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

window.addEventListener("error", (event) => {
  console.error("Global runtime error:", event.error ?? event.message);
});

window.addEventListener("unhandledrejection", (event) => {
  console.error("Unhandled promise rejection:", event.reason);
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RootErrorBoundary>
      <App />
    </RootErrorBoundary>
  </React.StrictMode>
);
