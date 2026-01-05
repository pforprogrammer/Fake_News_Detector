import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [backendStatus, setBackendStatus] = useState('checking');

  // Check backend connection on mount
  useEffect(() => {
    checkBackendConnection();
  }, []);

  const checkBackendConnection = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/health');
      if (response.ok) {
        const data = await response.json();
        setBackendStatus(data.model_trained ? 'connected' : 'no-model');
      } else {
        setBackendStatus('error');
      }
    } catch (err) {
      setBackendStatus('disconnected');
    }
  };

  const analyzeText = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);
    
    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to analyze text');
      }

      const data = await response.json();
      setResult(data);
      setBackendStatus('connected');
    } catch (err) {
      setError(`Error: ${err.message}. Make sure Python backend is running on http://localhost:5000`);
      setBackendStatus('disconnected');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      analyzeText();
    }
  };

  const exampleNews = [
    {
      title: "✅ Real Pakistan News Example",
      text: "According to a report by the State Bank of Pakistan, the country's foreign exchange reserves have increased by 2.3% this quarter. Economic analysts at Karachi University suggest this indicates improved trade balance."
    },
    {
      title: "❌ Fake Pakistan News Example",
      text: "SHOCKING!!! Pakistan's economy will COLLAPSE tomorrow! You won't believe what happens next! Share before government deletes this!!!"
    },
    {
      title: "✅ Educational News Example",
      text: "Research published in the Pakistan Journal of Medical Sciences shows promising results in treating dengue fever. The study was conducted at Aga Khan University Hospital over two years."
    },
    {
      title: "❌ Clickbait Example",
      text: "MIRACLE CURE discovered in Hunza Valley!!! Doctors are FURIOUS! This one weird trick will change Pakistan FOREVER!!!"
    }
  ];

  return (
    <div className="App">
      <div className="container">
        {/* Backend Status Indicator */}
        <div className={`backend-status ${backendStatus}`}>
          {backendStatus === 'checking' && '🔄 Checking backend connection...'}
          {backendStatus === 'connected' && '✅ Backend Connected & Model Ready'}
          {backendStatus === 'no-model' && '⚠️ Backend Connected but Model Not Trained'}
          {backendStatus === 'disconnected' && '❌ Backend Disconnected - Start Python server'}
          {backendStatus === 'error' && '⚠️ Backend Error'}
          {backendStatus === 'disconnected' && (
            <button onClick={checkBackendConnection} className="retry-btn">
              Retry Connection
            </button>
          )}
        </div>

        <header className="header">
          <h1>🇵🇰 Pakistan Fake News Detector</h1>
          <p>AI-powered news credibility analyzer with Machine Learning</p>
          <div className="tech-badges">
            <span className="badge">Python ML</span>
            <span className="badge">TF-IDF</span>
            <span className="badge">Naive Bayes</span>
            <span className="badge">React</span>
          </div>
        </header>

        <div className="main-content">
          <div className="input-section">
            <label htmlFor="newsText">
              Enter Pakistan news text to analyze:
              <span className="hint">Tip: Press Ctrl+Enter to analyze</span>
            </label>
            <textarea
              id="newsText"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Paste Pakistani news article text here... (e.g., from Dawn, Express Tribune, or any source)"
              rows="8"
            />
            
            <div className="button-group">
              <button 
                onClick={analyzeText} 
                disabled={loading || backendStatus === 'disconnected'}
                className="analyze-btn"
              >
                {loading ? '⏳ Analyzing with ML Model...' : '🚀 Analyze News'}
              </button>
              <button 
                onClick={() => {
                  setText('');
                  setResult(null);
                  setError('');
                }}
                className="clear-btn"
                disabled={loading}
              >
                🗑️ Clear
              </button>
            </div>

            {error && (
              <div className="error-message">
                ⚠️ {error}
              </div>
            )}
          </div>

          {result && (
            <div className={`result-section ${
              result.score >= 70 ? 'reliable' : 
              result.score >= 40 ? 'uncertain' : 'fake'
            }`}>
              <div className="result-header">
                <span className="result-icon">
                  {result.score >= 70 ? '✅' : result.score >= 40 ? '⚠️' : '❌'}
                </span>
                <div>
                  <h2>{result.classification}</h2>
                  <p>
                    Confidence: {result.confidence}% | Credibility Score: {result.score}/100
                  </p>
                  {result.ml_prediction !== undefined && (
                    <p className="ml-info">
                      🤖 ML Model Prediction: {result.ml_prediction === 1 ? 'Real News' : 'Fake News'} 
                      (Model Confidence: {result.model_confidence}%)
                    </p>
                  )}
                </div>
              </div>

              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ 
                    width: `${result.score}%`,
                    backgroundColor: result.score >= 70 ? '#10b981' : result.score >= 40 ? '#f59e0b' : '#ef4444'
                  }}
                />
              </div>

              {result.indicators && result.indicators.length > 0 && (
                <div className="indicators">
                  <h3>📊 Analysis Indicators:</h3>
                  <ul>
                    {result.indicators.map((indicator, idx) => (
                      <li key={idx} className={indicator.type}>
                        <span className="indicator-dot" />
                        {indicator.text}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="result-footer">
                <p>💡 Analysis powered by Machine Learning trained on Pakistan news dataset</p>
              </div>
            </div>
          )}

          <div className="examples-section">
            <h2>📄 Try Example Pakistan News</h2>
            <p className="examples-subtitle">
              These examples are based on Pakistan-specific news patterns
            </p>
            <div className="examples-grid">
              {exampleNews.map((example, idx) => (
                <div key={idx} className="example-card">
                  <h3>{example.title}</h3>
                  <p>{example.text}</p>
                  <button 
                    onClick={() => {
                      setText(example.text);
                      setResult(null);
                      setError('');
                    }}
                    className="use-example-btn"
                  >
                    Use this example →
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="info-section">
            <h3>🎓 How It Works</h3>
            <div className="info-grid">
              <div className="info-card">
                <div className="info-icon">🤖</div>
                <h4>Machine Learning</h4>
                <p>Uses Naive Bayes classifier trained on Pakistan news dataset</p>
              </div>
              <div className="info-card">
                <div className="info-icon">📊</div>
                <h4>TF-IDF Analysis</h4>
                <p>Analyzes word importance and patterns in news text</p>
              </div>
              <div className="info-card">
                <div className="info-icon">🇵🇰</div>
                <h4>Pakistan-Focused</h4>
                <p>Trained on Pakistani news sources and local patterns</p>
              </div>
              <div className="info-card">
                <div className="info-icon">⚡</div>
                <h4>Real-Time</h4>
                <p>Instant analysis with detailed credibility indicators</p>
              </div>
            </div>
          </div>
        </div>

        <footer className="footer">
          <p>🎓 College Project - Fake News Detection System</p>
          <p>Made with ❤️ for Pakistan | Python + Machine Learning + React</p>
          <p className="footer-note">
            ⚠️ Note: This is an educational project. Always verify news from multiple credible sources.
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;