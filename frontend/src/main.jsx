import React, { useState } from "react";
import ReactDOM from "react-dom/client";

import App from "./App.jsx";
import ExpertReview from "./ExpertReview.jsx";
import VerifiedArticles from "./VerifiedArticles.jsx";

import "./index.css";

function MainApp() {
  const [page, setPage] = useState("analysis");

  return (
    <div>
      <nav className="main-navigation">
        <div className="nav-title">
          🛡️ Fake News Detection
        </div>

        <div className="nav-buttons">
          <button
            className={
              page === "analysis"
                ? "active-nav"
                : ""
            }
            onClick={() => setPage("analysis")}
          >
            📰 User Analysis
          </button>

          <button
            className={
              page === "review"
                ? "active-nav"
                : ""
            }
            onClick={() => setPage("review")}
          >
            👨‍💼 Expert Review
          </button>

          <button
            className={
              page === "verified"
                ? "active-nav"
                : ""
            }
            onClick={() => setPage("verified")}
          >
            ✅ Verified Articles
          </button>
        </div>
      </nav>

      {page === "analysis" && <App />}

      {page === "review" && <ExpertReview />}

      {page === "verified" && <VerifiedArticles />}
    </div>
  );
}

ReactDOM.createRoot(
  document.getElementById("root")
).render(
  <React.StrictMode>
    <MainApp />
  </React.StrictMode>
);