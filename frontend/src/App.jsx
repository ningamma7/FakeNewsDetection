import { useEffect, useState } from 'react';
import { api, jsonBody } from './api.js';
import { ShieldCheck, ShieldAlert, LoaderCircle, Send, CheckCircle2, AlertTriangle, Search, ClipboardCheck, BookOpenCheck, RefreshCw, ExternalLink, Database } from 'lucide-react';
import './App.css';

const tabs = [
  { id: 'analysis', label: 'User Analysis', icon: Search },
  { id: 'review', label: 'Expert Review', icon: ClipboardCheck },
  { id: 'verified', label: 'Verified Articles', icon: BookOpenCheck },
];

export default function App() {
  const [tab, setTab] = useState('analysis');
  const [article, setArticle] = useState('');
  const [title, setTitle] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [suggestedLabel, setSuggestedLabel] = useState('');
  const [comment, setComment] = useState('');
  const [feedbackMessage, setFeedbackMessage] = useState('');
  const [reviewerToken, setReviewerToken] = useState('');
  const [pending, setPending] = useState([]);
  const [verified, setVerified] = useState([]);
  const [portalMessage, setPortalMessage] = useState('');
  const [portalLoading, setPortalLoading] = useState(false);
  const [reviewDrafts, setReviewDrafts] = useState({});

  useEffect(() => { api('/api/health').then(setHealth).catch(() => setHealth({ status: 'unavailable' })); }, []);

  async function analyze(e) {
    e.preventDefault();
    if (article.trim().length < 20) { setError('Enter at least 20 characters of article text.'); return; }
    setLoading(true); setError(''); setResult(null); setFeedbackMessage('');
    try {
      const data = await api('/api/analyze', { method: 'POST', body: jsonBody({ title: title.trim() || null, article: article.trim(), search_query: searchQuery.trim() || null }) });
      setResult(data);
    } catch (err) { setError(err.message || 'Analysis failed. Check that the backend and a model are available.'); }
    finally { setLoading(false); }
  }

  async function submitFeedback() {
    if (!result?.prediction) return;
    setError(''); setFeedbackMessage('');
    try {
      const data = await api('/api/feedback', { method: 'POST', body: jsonBody({
        article: `${title.trim() ? title.trim() + '\n\n' : ''}${article.trim()}`,
        prediction: result.prediction.prediction, confidence: result.prediction.confidence,
        suggested_label: suggestedLabel || null, comment: comment.trim() || null
      }) });
      setFeedbackMessage(`Feedback submitted for expert review (reference #${data.feedback_id}).`);
      setSuggestedLabel(''); setComment('');
    } catch (err) { setError(err.message || 'Could not submit feedback.'); }
  }

  async function loadReviewerData() {
    setPortalLoading(true); setPortalMessage('');
    try {
      const headers = { 'X-Reviewer-Token': reviewerToken };
      const [pendingData, verifiedData] = await Promise.all([
        api('/api/review/pending', { headers }), api('/api/verified')
      ]);
      setPending(pendingData.items || []); setVerified(verifiedData.items || []);
      setPortalMessage('Reviewer queue and verified articles refreshed.');
    } catch (err) { setPortalMessage(err.message || 'Unable to load reviewer queue. Check the reviewer token and .env configuration.'); }
    finally { setPortalLoading(false); }
  }

  function updateDraft(id, key, value) {
    setReviewDrafts(old => ({ ...old, [id]: { ...(old[id] || {}), [key]: value } }));
  }

  async function reviewItem(item, action) {
    const draft = reviewDrafts[item.id] || {};
    setPortalMessage('');
    try {
      const payload = {
        action,
        final_label: action === 'approve' ? (draft.final_label || item.user_suggested_label || 'True News') : null,
        evidence_url: draft.evidence_url || '',
        notes: draft.notes || '',
        publication_date: draft.publication_date || null,
      };
      await api(`/api/review/${item.id}`, { method: 'POST', headers: { 'X-Reviewer-Token': reviewerToken }, body: jsonBody(payload) });
      setPortalMessage(`Feedback #${item.id} ${action === 'approve' ? 'approved and published' : 'rejected'}.`);
      await loadReviewerData();
    } catch (err) { setPortalMessage(err.message || 'Could not save this review.'); }
  }

  const prediction = result?.prediction;
  const fake = prediction?.prediction === 'Fake News';
  const explanation = prediction?.explanation || {};
  const confidence = Number(prediction?.confidence || 0);
  const factChecks = result?.fact_checks || {};
  const news = result?.related_news || {};

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-icon"><ShieldCheck size={22}/></div><div><strong>FakeNews Detection</strong><span>AI-POWERED ARTICLE ANALYSIS</span></div></div>
      <div className="connection"><i className={health?.status === 'ok' ? 'online' : ''}/>{health?.status === 'ok' ? (health?.model?.loaded ? `Model ready · ${health.model.model_type}` : 'API connected · model needed') : 'Checking API connection'}</div>
    </header>
    <main className="page">
      <section className="hero"><div className="eyebrow">NEWS CLASSIFICATION & HUMAN VERIFICATION</div><h1>Understand a news article before sharing it.</h1><p>Get a machine-learning prediction, inspect the explanation, compare published fact-checks, and send uncertain cases for expert review. A model prediction is not proof of truth.</p></section>
      <nav className="main-tabs" aria-label="Main navigation">{tabs.map(({id,label,icon:Icon})=><button key={id} className={tab===id?'active':''} onClick={()=>{setTab(id);setError('');setPortalMessage('');}}><Icon size={16}/>{label}</button>)}</nav>
      {error && <div className="error"><AlertTriangle size={17}/>{error}</div>}
      {tab === 'analysis' && <>
        <div className="layout">
          <section className="panel input-panel">
            <div className="panel-heading"><span className="step">01</span><div><h2>Article details</h2><p>Provide the headline and article text.</p></div></div>
            <form onSubmit={analyze}>
              <label>Headline <span>Optional</span><input value={title} onChange={e=>setTitle(e.target.value)} placeholder="Enter the news headline"/></label>
              <label>Article text<textarea value={article} onChange={e=>setArticle(e.target.value)} rows={11} placeholder="Paste the full news article here..." required/></label>
              <label>Fact-check search query <span>Optional</span><input value={searchQuery} onChange={e=>setSearchQuery(e.target.value)} placeholder="A concise claim to search for"/></label>
              <div className="form-bottom"><span>{article.trim().length.toLocaleString()} characters</span><button className="primary" type="submit" disabled={loading}>{loading ? <><LoaderCircle className="spin" size={17}/> Analyzing…</> : <><ShieldCheck size={17}/> Analyze article</>}</button></div>
            </form>
            <p className="hint">The model learns patterns from its training data. It does not independently establish factual truth.</p>
          </section>
          <section className="results">
            {!result && <div className="empty panel"><div className="empty-icon"><ShieldAlert size={27}/></div><h2>Your result will appear here</h2><p>Submit an article to see classification, confidence, Captum Saliency or TF-IDF explanation, and fact-check lookup status.</p></div>}
            {result && <>
              <section className={`verdict panel ${fake ? 'fake' : 'true'}`}>
                <div className="panel-heading"><span className="step">02</span><div><h2>Prediction result</h2><p>Machine-learning classification for the submitted text.</p></div><span className={`pill ${fake?'red':'green'}`}>{prediction?.prediction || 'Unknown'}</span></div>
                <div className="confidence"><strong>{(confidence*100).toFixed(1)}%</strong><span>model confidence · {prediction?.model_type || 'model'}</span></div>
                <div className="bar-row"><span>Fake News</span><div className="bar"><i className="bar-fake" style={{width:`${100*(prediction?.probabilities?.['Fake News'] || 0)}%`}}/></div><b>{(100*(prediction?.probabilities?.['Fake News'] || 0)).toFixed(1)}%</b></div>
                <div className="bar-row"><span>True News</span><div className="bar"><i className="bar-true" style={{width:`${100*(prediction?.probabilities?.['True News'] || 0)}%`}}/></div><b>{(100*(prediction?.probabilities?.['True News'] || 0)).toFixed(1)}%</b></div>
                {prediction?.needs_human_review && <div className="caution"><AlertTriangle size={16}/> Low confidence: consider human review before relying on this result.</div>}
                <p className="disclaimer">{prediction?.disclaimer || result.disclaimer}</p>
              </section>
              <section className="panel">
                <div className="panel-heading"><span className="step">03</span><div><h2>Why this result?</h2><p>Important tokens identified by the model explanation.</p></div><span className="method">{explanation.method || 'Explanation'}</span></div>
                {explanation.status === 'available' && explanation.features?.length ? <div className="tokens">{explanation.features.map((f,i)=><span className="token" key={`${f.token}-${i}`}>{f.token}<small>{Math.round((f.importance||0)*100)}%</small></span>)}</div> : <p className="muted">{explanation.note || 'Token explanation is unavailable for this prediction.'}</p>}
                <p className="hint">Saliency/feature weights show model sensitivity, not proof that the article is true or false.</p>
              </section>
              <section className="panel">
                <div className="panel-heading"><span className="step">04</span><div><h2>Published fact-checks</h2><p>Google ClaimReview matches are independent of the model prediction.</p></div></div>
                <div className="status-line"><span className={`status-dot ${factChecks.status==='matches_found'?'good':''}`}/><strong>{factChecks.status==='matches_found' ? `${result.fact_check_review_count} published review(s) found` : factChecks.status==='api_key_not_configured' ? 'Google Fact Check API key not configured' : factChecks.status==='provider_error' ? 'Fact-check provider error' : 'No matching fact-check found'}</strong></div>
                {factChecks.note && <p className="muted">{factChecks.note}</p>}
                {(factChecks.results || []).map((claim,i)=><div className="evidence-item" key={`${claim.claim}-${i}`}><strong>{claim.claim || 'Claim review'}</strong>{claim.claimant && <p className="muted">Claimant: {claim.claimant}</p>}{(claim.reviews||[]).map((review,j)=><p className="review-result" key={`${review.url}-${j}`}><b>{review.rating || 'Rating not supplied'}</b> · {review.publisher || 'Publisher'} {review.title && `— ${review.title}`} {review.url && <a href={review.url} target="_blank" rel="noreferrer">Open review <ExternalLink size={12}/></a>}</p>)}</div>)}
                <p className="hint">No matching fact-check does not mean the claim is true. Read the publisher's review and context.</p>
              </section>
              <section className="panel">
                <div className="panel-heading"><span className="step">05</span><div><h2>Related news coverage</h2><p>Search results are context, not verification.</p></div></div>
                {news.error && <p className="muted">{news.error}</p>}
                {(news.results || []).length ? (news.results||[]).map((item,i)=><div className="evidence-item" key={`${item.url}-${i}`}><a href={item.url} target="_blank" rel="noreferrer">{item.title || item.url} <ExternalLink size={12}/></a><p className="muted">{item.domain} {item.published_at && `· ${item.published_at}`}</p></div>) : <p className="muted">No related news items returned. This is not evidence for either label.</p>}
              </section>
              <section className="panel">
                <div className="panel-heading"><span className="step">06</span><div><h2>Submit for expert review</h2><p>Suggest a correction and add a source if you have one.</p></div></div>
                <label>Your suggested label<select value={suggestedLabel} onChange={e=>setSuggestedLabel(e.target.value)}><option value="">No suggested label</option><option value="True News">True News</option><option value="Fake News">Fake News</option></select></label>
                <label>Comment / supporting source<textarea rows={3} value={comment} onChange={e=>setComment(e.target.value)} placeholder="Optional explanation or source URL"/></label>
                <div className="form-bottom"><span>Submissions remain pending until reviewed.</span><button className="secondary" onClick={submitFeedback} disabled={!suggestedLabel && !comment.trim()}><Send size={16}/> Submit feedback</button></div>
                {feedbackMessage && <p className="success"><CheckCircle2 size={16}/>{feedbackMessage}</p>}
              </section>
            </>}
          </section>
        </div>
      </>}
      {tab === 'review' && <section className="panel portal-panel">
        <div className="panel-heading"><span className="step"><ClipboardCheck size={16}/></span><div><h2>Expert Review Portal</h2><p>Restricted actions require the reviewer token configured in the backend .env file.</p></div></div>
        <div className="reviewer-login"><label>Reviewer access token<input type="password" value={reviewerToken} onChange={e=>setReviewerToken(e.target.value)} placeholder="Enter REVIEWER_TOKEN" autoComplete="off"/></label><button className="primary" onClick={loadReviewerData} disabled={portalLoading || !reviewerToken}>{portalLoading ? <LoaderCircle className="spin" size={16}/> : <RefreshCw size={16}/>} Load review queue</button></div>
        {portalMessage && <p className="portal-message">{portalMessage}</p>}
        <div className="section-title"><h3>Pending submissions</h3><span>{pending.length} item(s)</span></div>
        {!pending.length && <p className="muted">No pending items loaded. Enter the reviewer token and refresh the queue.</p>}
        {pending.map(item=>{const draft=reviewDrafts[item.id]||{};return <article className="review-card" key={item.id}>
          <div className="review-card-top"><strong>Feedback #{item.id}</strong><span className="pill amber">Pending</span></div>
          <p className="article-preview">{item.article}</p>
          <div className="meta-grid"><span>Model: <b>{item.prediction_label}</b></span><span>Confidence: <b>{(Number(item.confidence)*100).toFixed(1)}%</b></span><span>Suggested: <b>{item.user_suggested_label||'None'}</b></span><span>Submitted: <b>{item.created_at}</b></span></div>
          {item.user_comment && <p className="review-comment">User comment: {item.user_comment}</p>}
          <div className="review-fields"><label>Final label<select value={draft.final_label ?? item.user_suggested_label ?? 'True News'} onChange={e=>updateDraft(item.id,'final_label',e.target.value)}><option>Fake News</option><option>True News</option></select></label><label>Evidence URL (required to approve)<input value={draft.evidence_url||''} onChange={e=>updateDraft(item.id,'evidence_url',e.target.value)} placeholder="https://credible-source.example/review"/></label><label>Review notes<textarea rows={2} value={draft.notes||''} onChange={e=>updateDraft(item.id,'notes',e.target.value)} placeholder="Evidence assessment and rationale"/></label><label>Publication date (optional)<input value={draft.publication_date||''} onChange={e=>updateDraft(item.id,'publication_date',e.target.value)} placeholder="YYYY-MM-DD"/></label></div>
          <div className="form-bottom"><span>Approval requires a final label and http(s) evidence URL.</span><div className="action-row"><button className="secondary reject" onClick={()=>reviewItem(item,'reject')}><AlertTriangle size={14}/> Reject</button><button className="primary" onClick={()=>reviewItem(item,'approve')}><CheckCircle2 size={14}/> Approve & publish</button></div></div>
        </article>})}
      </section>}
      {tab === 'verified' && <section className="panel portal-panel">
        <div className="panel-heading"><span className="step"><BookOpenCheck size={16}/></span><div><h2>Verified Articles</h2><p>Only items explicitly approved by an authorized reviewer appear here.</p></div><button className="secondary refresh-btn" onClick={()=>api('/api/verified').then(d=>setVerified(d.items||[])).catch(e=>setPortalMessage(e.message))}><RefreshCw size={14}/> Refresh</button></div>
        {portalMessage && <p className="portal-message">{portalMessage}</p>}
        {!verified.length && <div className="empty"><div className="empty-icon"><Database size={25}/></div><h2>No approved articles yet</h2><p>Reviewed and approved articles with evidence links will appear here.</p></div>}
        {verified.map(item=><article className="verified-card" key={item.id}><div className="review-card-top"><strong>{item.label===0?'Fake News':'True News'}</strong><span className="pill green">Reviewer approved</span></div><p className="article-preview">{item.article}</p><div className="meta-grid"><span>Approved: <b>{item.created_at}</b></span><span>Label available at: <b>{item.label_available_at}</b></span></div>{item.reviewer_notes&&<p className="review-comment">Reviewer notes: {item.reviewer_notes}</p>}<a href={item.evidence_url} target="_blank" rel="noreferrer">Open evidence source <ExternalLink size={13}/></a></article>)}
      </section>}
      <footer>FakeNews Detection Platform · SQLite stores prediction history, fact-check lookups, review submissions and approved articles. Never treat a model score as proof.</footer>
    </main>
  </div>;
}
