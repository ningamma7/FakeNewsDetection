import { useEffect, useState } from "react";
import axios from "axios";
import {
  ShieldCheck,
  AlertTriangle,
  Loader2,
} from "lucide-react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [article, setArticle] = useState("");
  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Captum explanation
  const [explanation, setExplanation] = useState(null);
  const [explanationLoading, setExplanationLoading] =
    useState(false);
  const [explanationError, setExplanationError] =
    useState("");

  // Feedback
  const [feedbackLoading, setFeedbackLoading] =
    useState(false);
  const [feedbackMessage, setFeedbackMessage] =
    useState("");

  // Analysis history
  const [history, setHistory] = useState(() => {
    try {
      const savedHistory =
        localStorage.getItem("analysisHistory");

      return savedHistory
        ? JSON.parse(savedHistory)
        : [];
    } catch (error) {
      console.error(
        "Unable to load analysis history:",
        error
      );

      return [];
    }
  });

  // --------------------------------------------------
  // Save history to localStorage
  // --------------------------------------------------

  useEffect(() => {
    try {
      localStorage.setItem(
        "analysisHistory",
        JSON.stringify(history)
      );
    } catch (error) {
      console.error(
        "Unable to save analysis history:",
        error
      );
    }
  }, [history]);

  // --------------------------------------------------
  // Submit human feedback
  // --------------------------------------------------

  const submitFeedback = async (verifiedLabel) => {
    if (!result || !article.trim()) {
      return;
    }

    setFeedbackLoading(true);
    setFeedbackMessage("");

    try {
      await axios.post(
        `${API_URL}/feedback`,
        {
          article: article,
          prediction: result.prediction,
          confidence: result.confidence_score,
          verified_label: verifiedLabel,
        }
      );

      setFeedbackMessage(
        "Feedback saved successfully."
      );
    } catch (error) {
      console.error(
        "Feedback error:",
        error
      );

      setFeedbackMessage(
        "Unable to save feedback. Please make sure the backend is running."
      );
    } finally {
      setFeedbackLoading(false);
    }
  };

  // --------------------------------------------------
  // Get Captum Saliency explanation
  // --------------------------------------------------

  const getExplanation = async (articleText) => {
    setExplanationLoading(true);
    setExplanationError("");
    setExplanation(null);

    try {
      const response = await fetch(
        `${API_URL}/explain`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            article: articleText,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Explanation backend returned status ${response.status}`
        );
      }

      const data = await response.json();

      console.log(
        "Captum Saliency explanation:",
        data
      );

      setExplanation(data);
    } catch (err) {
      console.error(
        "Explanation error:",
        err
      );

      setExplanationError(
        "Unable to generate explanation. Please try again."
      );
    } finally {
      setExplanationLoading(false);
    }
  };

  // --------------------------------------------------
  // Analyze news article
  // --------------------------------------------------

  const analyzeNews = async () => {
    if (!article.trim()) {
      setError(
        "Please enter a news article."
      );
      return;
    }

    setLoading(true);
    setError("");

    setResult(null);
    setExplanation(null);
    setExplanationError("");
    setFeedbackMessage("");

    try {
      // ----------------------------------------------
      // Step 1: Prediction
      // ----------------------------------------------

      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            article: article.trim(),
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Backend returned status ${response.status}`
        );
      }

      const data = await response.json();

      console.log(
        "Backend prediction:",
        data
      );

      setResult(data);

      // ----------------------------------------------
      // Step 2: Captum Saliency explanation
      // ----------------------------------------------

      await getExplanation(
        article.trim()
      );

      // ----------------------------------------------
      // Step 3: Save analysis history
      // ----------------------------------------------

      setHistory(
        (previousHistory) => [
          {
            id: Date.now(),
            article: article.trim(),
            prediction: data.prediction,
            confidence_score:
              data.confidence_score,
            confidence_level:
              data.confidence_level,
            time: new Date().toLocaleString(),
          },
          ...previousHistory,
        ]
      );
    } catch (err) {
      console.error(
        "Connection error:",
        err
      );

      setError(
        `Unable to connect to backend. ${
          err.message || "Please check the server."
        }`
      );
    } finally {
      setLoading(false);
    }
  };

  // --------------------------------------------------
  // Clear current analysis
  // --------------------------------------------------

  const clearAll = () => {
    setArticle("");
    setResult(null);
    setError("");

    setExplanation(null);
    setExplanationError("");

    setFeedbackMessage("");
  };

  // --------------------------------------------------
  // Clear analysis history
  // --------------------------------------------------

  const clearHistory = () => {
    setHistory([]);
  };

  // --------------------------------------------------
  // Calculate average confidence safely
  // --------------------------------------------------

  const averageConfidence =
    history.length > 0
      ? (
          history.reduce(
            (total, item) =>
              total +
              (Number(item.confidence_score) || 0),
            0
          ) / history.length
        ).toFixed(2)
      : "0.00";

  // --------------------------------------------------
  // Render
  // --------------------------------------------------

  return (
    <div className="app-container">

      {/* ==================================================
          HEADER
      ================================================== */}

      <header className="header">
        <div className="brand">

          <ShieldCheck size={36} />

          <div>
            <h1>
              Fake News Detection
            </h1>

            <p>
              AI-powered news analysis
            </p>
          </div>

        </div>
      </header>


      {/* ==================================================
          MAIN CONTENT
      ================================================== */}

      <main className="main-content">

        {/* --------------------------------------------------
            INTRODUCTION
        -------------------------------------------------- */}

        <section className="intro-section">

          <h2>
            Analyze a News Article
          </h2>

          <p>
            Enter a news article below to receive
            a machine-learning prediction,
            confidence score, and explainable
            AI insights.
          </p>

        </section>


        {/* --------------------------------------------------
            ARTICLE INPUT
        -------------------------------------------------- */}

        <section className="analysis-card">

          <label htmlFor="article">
            News Article
          </label>

          <textarea
            id="article"
            placeholder="Paste or type your news article here..."
            value={article}
            onChange={(e) =>
              setArticle(e.target.value)
            }
            disabled={loading}
          />

          <div className="button-group">

            <button
              className="analyze-button"
              onClick={analyzeNews}
              disabled={
                loading ||
                !article.trim()
              }
            >

              {loading ? (
                <>
                  <Loader2
                    className="spin"
                    size={20}
                  />

                  Analyzing...
                </>
              ) : (
                <>
                  <ShieldCheck
                    size={20}
                  />

                  Analyze News
                </>
              )}

            </button>


            <button
              className="clear-button"
              onClick={clearAll}
              disabled={loading}
            >
              Clear
            </button>

          </div>


          {/* Error */}

          {error && (
            <div className="error-message">

              <AlertTriangle
                size={20}
              />

              {error}

            </div>
          )}

        </section>


        {/* ==================================================
            PREDICTION RESULT
        ================================================== */}

        {result && (
          <section className="result-card">

            <h2>
              Analysis Result
            </h2>


            {/* ------------------------------------------------
                Prediction
            ------------------------------------------------ */}

            <div
              className={`prediction ${
                result.prediction ===
                "Fake News"
                  ? "fake"
                  : "true"
              }`}
            >

              <div>

                <h3>
                  {result.prediction}
                </h3>

                <p>
                  {result.confidence_level}
                </p>

              </div>

            </div>


            {/* ------------------------------------------------
                Confidence
            ------------------------------------------------ */}

            <div className="result-details">

              <div className="detail-item">

                <span>
                  Confidence Score
                </span>

                <strong>
                  {Number(
                    result.confidence_score
                  ).toFixed(2)}
                  %
                </strong>

              </div>


              <div className="confidence-bar">

                <div
                  className="confidence-fill"
                  style={{
                    width: `${Math.min(
                      Math.max(
                        Number(
                          result.confidence_score
                        ) || 0,
                        0
                      ),
                      100
                    )}%`,
                  }}
                />

              </div>


              {/* Recommendation */}

              <div className="recommendation">

                <strong>
                  Recommendation:
                </strong>

                <p>
                  {result.recommendation}
                </p>

              </div>

            </div>


            {/* ==================================================
                CAPTUM SALIENCY EXPLANATION
            ================================================== */}

            <div className="explanation-box">

              <h3>
                Why this result?
              </h3>

              <p>
                Captum Saliency identifies the
                words that had the strongest
                influence on the model's
                prediction.
              </p>


              {/* Explanation Loading */}

              {explanationLoading && (
                <div className="explanation-loading">

                  <Loader2
                    className="spin"
                    size={18}
                  />

                  Generating explanation...

                </div>
              )}


              {/* Explanation Error */}

              {explanationError && (
                <p className="explanation-error">
                  {explanationError}
                </p>
              )}


              {/* Explanation Results */}

              {explanation &&
                explanation.top_features && (
                  <>

                    {/* Model Information */}

                    <div className="explanation-method">

                      <span>
                        <strong>
                          Model:
                        </strong>{" "}
                        DistilBERT
                      </span>

                      <span>
                        <strong>
                          Method:
                        </strong>{" "}
                        Captum Saliency
                      </span>

                    </div>


                    {/* Important Words */}

                    <h4>
                      Important Words
                    </h4>


                    <div className="feature-list">

                      {explanation.top_features
                        .map(
                          (
                            feature,
                            index
                          ) => {

                            const score =
                              Number(
                                feature.importance
                              ) || 0;

                            const magnitude =
                              Math.min(
                                Math.abs(
                                  score
                                ) * 100,
                                100
                              );

                            return (
                              <div
                                key={`${feature.token}-${index}`}
                                className="feature-item supports"
                              >

                                {/* Feature Header */}

                                <div className="feature-header">

                                  <span className="feature-token">
                                    {
                                      feature.token
                                    }
                                  </span>

                                  <span className="feature-score">
                                    {score.toFixed(
                                      2
                                    )}
                                  </span>

                                </div>


                                {/* Importance Bar */}

                                <div className="feature-bar-container">

                                  <div
                                    className="feature-bar"
                                    style={{
                                      width: `${magnitude}%`,
                                    }}
                                  />

                                </div>


                                {/* Feature Meaning */}

                                <div className="feature-effect">
                                  Important for prediction
                                </div>

                              </div>
                            );
                          }
                        )}

                    </div>


                    {/* Explanation Note */}

                    <p className="explanation-note">

                      Higher importance indicates
                      that a word had a stronger
                      influence on the model's
                      prediction.

                      These explanations describe
                      model behavior and do not prove
                      that an article is factually
                      true.

                    </p>

                  </>
                )}


              {/* No Explanation */}

              {!explanationLoading &&
                !explanation &&
                !explanationError && (
                  <p>
                    Explanation will appear
                    after analysis.
                  </p>
                )}

            </div>


            {/* ==================================================
                HUMAN FEEDBACK
            ================================================== */}

            <div className="verification-section">

              <h3>
                Report / Provide Feedback
              </h3>

              <p>
                If you believe the ML model
                prediction is incorrect, or you
                have additional information about
                this article, provide your feedback.
                An authorized reviewer can verify
                the submission later.
              </p>


              <div className="verification-buttons">

                {/* Fake */}

                <button
                  className="verify-fake-button"
                  onClick={() =>
                    submitFeedback(
                      "Fake News"
                    )
                  }
                  disabled={feedbackLoading}
                >

                  {feedbackLoading ? (
                    <Loader2
                      className="spin"
                      size={16}
                    />
                  ) : null}

                  I Believe This Is Fake

                </button>


                {/* True */}

                <button
                  className="verify-true-button"
                  onClick={() =>
                    submitFeedback(
                      "True News"
                    )
                  }
                  disabled={feedbackLoading}
                >

                  {feedbackLoading ? (
                    <Loader2
                      className="spin"
                      size={16}
                    />
                  ) : null}

                  I Believe This Is True

                </button>


                {/* Unknown */}

                <button
                  className="verify-unknown-button"
                  onClick={() =>
                    submitFeedback(
                      "Unverified"
                    )
                  }
                  disabled={feedbackLoading}
                >

                  {feedbackLoading ? (
                    <Loader2
                      className="spin"
                      size={16}
                    />
                  ) : null}

                  I Cannot Verify

                </button>

              </div>


              {/* Feedback Message */}

              {feedbackMessage && (
                <p className="feedback-message">
                  {feedbackMessage}
                </p>
              )}

            </div>


            {/* ==================================================
                DISCLAIMER
            ================================================== */}

            <p className="disclaimer">

              This prediction is generated from
              learned text patterns and does not
              independently verify factual claims.

              Captum explanations describe model
              behavior and are not proof of factual
              accuracy.

            </p>

          </section>
        )}

      </main>


      {/* ==================================================
          STATISTICS
      ================================================== */}

      {history.length > 0 && (
        <section className="stats-grid">

          {/* Total */}

          <div className="stat-card">

            <h3>
              Total Analyzed
            </h3>

            <strong>
              {history.length}
            </strong>

          </div>


          {/* Fake */}

          <div className="stat-card fake-stat">

            <h3>
              Fake News
            </h3>

            <strong>
              {
                history.filter(
                  (item) =>
                    item.prediction ===
                    "Fake News"
                ).length
              }
            </strong>

          </div>


          {/* True */}

          <div className="stat-card true-stat">

            <h3>
              True News
            </h3>

            <strong>
              {
                history.filter(
                  (item) =>
                    item.prediction ===
                    "True News"
                ).length
              }
            </strong>

          </div>


          {/* Average Confidence */}

          <div className="stat-card">

            <h3>
              Average Confidence
            </h3>

            <strong>
              {averageConfidence}%
            </strong>

          </div>

        </section>
      )}


      {/* ==================================================
          ANALYSIS HISTORY
      ================================================== */}

      {history.length > 0 && (
        <section className="history-card">

          <div className="history-header">

            <h2>
              Analysis History
            </h2>

            <button
              className="clear-history-button"
              onClick={clearHistory}
            >
              Clear History
            </button>

          </div>


          {/* History Items */}

          {history.map(
            (item, index) => (
              <div
                className="history-item"
                key={
                  item.id || index
                }
              >

                <p className="history-article">
                  {item.article}
                </p>


                <div className="history-info">

                  <strong>
                    {item.prediction}
                  </strong>

                  <span>
                    {Number(
                      item.confidence_score
                    ).toFixed(2)}
                    % confidence
                  </span>

                  <small>
                    {item.time}
                  </small>

                </div>

              </div>
            )
          )}

        </section>
      )}


      {/* ==================================================
          FOOTER
      ================================================== */}

      <footer>

        <p>
          Fake News Detection Project |
          Machine Learning and NLP
        </p>

      </footer>

    </div>
  );
}

export default App;