import { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function VerifiedArticles() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const fetchVerifiedArticles = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await axios.get(
        `${API_URL}/verified`
      );

      setRecords(response.data.records || []);
    } catch (err) {
      console.error(err);
      setError("Unable to load verified articles.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchVerifiedArticles();
  }, []);

  return (
    <div className="expert-container">
      <div className="expert-header">
        <div>
          <h1>Verified Articles</h1>
          <p>
            Articles approved through the expert review process.
          </p>
        </div>

        <div className="pending-count">
          Verified Articles: {records.length}
        </div>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {loading ? (
        <div className="review-message">
          Loading verified articles...
        </div>
      ) : records.length === 0 ? (
        <div className="review-message">
          <h2>No Verified Articles</h2>
          <p>
            Approved articles will appear here after expert review.
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
                  Verification ID: {record.id}
                </span>

                <span className="approved-label">
                  {record.verification_status}
                </span>
              </div>

              <h3>Article</h3>

              <div className="article-preview">
                {record.text}
              </div>

              <div className="review-details">
                <div>
                  <strong>Final Verified Label</strong>
                  <span>{record.final_label}</span>
                </div>

                <div>
                  <strong>Original Prediction</strong>
                  <span>{record.original_prediction}</span>
                </div>

                <div>
                  <strong>Original Confidence</strong>
                  <span>
                    {record.original_confidence}%
                  </span>
                </div>

                <div>
                  <strong>Training Status</strong>
                  <span>{record.training_status}</span>
                </div>

                <div>
                  <strong>Review Date</strong>
                  <span>{record.date}</span>
                </div>
              </div>

              <div className="full-article">
                <strong>Expert Notes</strong>
                <p>
                  {record.expert_notes ||
                    "No expert notes provided."}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default VerifiedArticles;