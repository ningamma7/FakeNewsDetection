import { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function ExpertReview() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [selectedRecord, setSelectedRecord] = useState(null);

  const [decision, setDecision] = useState("Approve");
  const [finalLabel, setFinalLabel] = useState("Fake News");
  const [expertNotes, setExpertNotes] = useState("");

  const fetchPendingReviews = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await axios.get(
        `${API_URL}/feedback/pending`
      );

      setRecords(response.data.records || []);
    } catch (err) {
      console.error(err);
      setError("Unable to load pending reviews.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPendingReviews();
  }, []);

  const openReview = (record) => {
    setSelectedRecord(record);

    setDecision("Approve");

    setFinalLabel(
      record.prediction === "Fake News"
        ? "Fake News"
        : "True News"
    );

    setExpertNotes("");
  };

  const closeReview = () => {
    setSelectedRecord(null);
    setExpertNotes("");
  };

  const submitReview = async (event) => {
    event.preventDefault();

    if (!selectedRecord) {
      return;
    }

    try {
      const response = await axios.post(
        `${API_URL}/feedback/${selectedRecord.id}/review`,
        {
          decision: decision,
          final_label: finalLabel,
          expert_notes: expertNotes,
        }
      );

      alert(
        `Review completed successfully.\n\n` +
        `Status: ${response.data.verification_status}\n` +
        `Final Label: ${response.data.final_label || "None"}`
      );

      closeReview();

      fetchPendingReviews();

    } catch (err) {
      console.error("Review error:", err);

      const message =
        err.response?.data?.detail ||
        "Unable to submit expert review.";

      alert(message);
    }
  };

  return (
    <div className="expert-container">

      <div className="expert-header">
        <div>
          <h1>Expert Review Dashboard</h1>

          <p>
            Review selected user feedback and record
            the final verified classification.
          </p>
        </div>

        <div className="pending-count">
          Pending Reviews: {records.length}
        </div>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {loading ? (
        <div className="review-message">
          Loading pending reviews...
        </div>
      ) : records.length === 0 ? (
        <div className="review-message">
          <h2>No Pending Reviews</h2>

          <p>
            There are currently no articles waiting
            for review.
          </p>
        </div>
      ) : (
        <div className="review-list">

          {records.map((record) => (

            <div
              className="review-card"
              key={record.id}
            >

              <div className="review-card-header">

                <span>
                  Review ID: {record.id}
                </span>

                <span className="pending-label">
                  Pending
                </span>

              </div>

              <h3>Article</h3>

              <div className="article-preview">
                {record.text}
              </div>

              <div className="review-details">

                <div>
                  <strong>
                    ML Model Prediction
                  </strong>

                  <span>
                    {record.prediction}
                  </span>
                </div>

                <div>
                  <strong>
                    Confidence
                  </strong>

                  <span>
                    {record.confidence}%
                  </span>
                </div>

                <div>
                  <strong>
                    User Feedback
                  </strong>

                  <span>
                    {record.verified_label}
                  </span>
                </div>

              </div>

              <button
                className="review-button"
                onClick={() => openReview(record)}
              >
                Review Article
              </button>

            </div>

          ))}

        </div>
      )}

      {selectedRecord && (

        <div className="review-modal">

          <div className="review-modal-content">

            <h2>
              Complete Expert Review
            </h2>

            <div className="full-article">

              <h3>Article</h3>

              <p>
                {selectedRecord.text}
              </p>

            </div>

            <div className="review-information">

              <p>
                <strong>
                  ML Model Prediction:
                </strong>{" "}
                {selectedRecord.prediction}
              </p>

              <p>
                <strong>
                  ML Confidence:
                </strong>{" "}
                {selectedRecord.confidence}%
              </p>

              <p>
                <strong>
                  User Feedback:
                </strong>{" "}
                {selectedRecord.verified_label}
              </p>

            </div>

            <form onSubmit={submitReview}>

              <label>
                Review Decision
              </label>

              <select
                value={decision}
                onChange={(event) =>
                  setDecision(event.target.value)
                }
              >

                <option value="Approve">
                  Approve
                </option>

                <option value="Reject">
                  Reject
                </option>

              </select>

              <label>
                Final Verified Label
              </label>

              <select
                value={finalLabel}
                onChange={(event) =>
                  setFinalLabel(event.target.value)
                }
              >

                <option value="Fake News">
                  Fake News
                </option>

                <option value="True News">
                  True News
                </option>

              </select>

              <label>
                Expert Notes
              </label>

              <textarea
                value={expertNotes}
                onChange={(event) =>
                  setExpertNotes(event.target.value)
                }
                placeholder="Enter the reason for your decision..."
                rows="5"
              />

              <div className="review-actions">

                <button
                  type="submit"
                  className="submit-review-button"
                >
                  Submit Review
                </button>

                <button
                  type="button"
                  className="cancel-review-button"
                  onClick={closeReview}
                >
                  Cancel
                </button>

              </div>

            </form>

          </div>

        </div>

      )}

    </div>
  );
}

export default ExpertReview;