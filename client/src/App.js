import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [queryType, setQueryType] = useState(null);
  const [searchHistory, setSearchHistory] = useState([]);
  const chatContainerRef = useRef(null);
  
  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    setLoading(true);
    setResponse(null);
    
    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      
      const data = await res.json();
      setResponse(data);
      setQueryType(data.query_type);
      
      // Add to search history
      setSearchHistory(prev => [
        { query, response: data, timestamp: new Date().toLocaleTimeString() },
        ...prev.slice(0, 9)
      ]);
    } catch (error) {
      console.error('Error fetching data:', error);
      setResponse({
        query_type: 'ERROR',
        content: 'An error occurred while processing your request. Please try again.'
      });
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [response]);
  
  // Format text with markdown-like syntax (* for italics, ** for bold)
  const formatText = (text) => {
    if (!text) return '';
    
    // Replace **text** with <strong>text</strong>
    let formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Replace *text* with <em>text</em>
    formattedText = formattedText.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    return formattedText;
  };
  
  const renderResponse = () => {
    if (!response) return null;
    
    const typeLabels = {
      'GENERAL': 'Clinical Guidance',
      'CASE': 'Case Analysis',
      'PREDICTION': 'Clinical Prediction',
      'ERROR': 'Error'
    };
    
    return (
      <div className="response-container">
        <div className="response-header">
          <span className={`response-type ${response.query_type.toLowerCase()}`}>
            {typeLabels[response.query_type] || response.query_type}
          </span>
        </div>
        
        {response.query_type === 'CASE' && response.similar_cases && (
          <div className="case-studies">
            <h3>Similar Case Studies</h3>
            {response.similar_cases.map((caseStudy, idx) => (
              <div key={idx} className="case-card">
                <div className="case-context">
                  <strong>Patient Context:</strong>
                  <p dangerouslySetInnerHTML={{ __html: formatText(caseStudy.context) }}></p>
                </div>
                <div className="case-response">
                  <strong>Clinical Approach:</strong>
                  <p dangerouslySetInnerHTML={{ __html: formatText(caseStudy.response) }}></p>
                </div>
              </div>
            ))}
          </div>
        )}
        
        <div className="response-content">
          {response.query_type === 'PREDICTION' ? (
            <div dangerouslySetInnerHTML={{ __html: response.content }} />
          ) : (
            <div dangerouslySetInnerHTML={{ __html: formatText(response.content) }}></div>
          )}
        </div>
        
        {response.sources && (
          <div className="sources">
            <h4>Sources</h4>
            <ul>
              {response.sources.map((source, idx) => (
                <li key={idx}>{source}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>LegacySearch</h1>
        <p className="app-subtitle">Mental Health Guidance for Clinicians</p>
        <div className="search-container">
          <form onSubmit={handleSearch}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask about patient cases, treatments, or clinical predictions..."
              className="search-input"
            />
            <button type="submit" className="search-button">
              {loading ? "Analyzing..." : "Search"}
            </button>
          </form>
        </div>
      </header>

      <main className="app-main">
        <div className="chat-container" ref={chatContainerRef}>
          {!response && !loading && (
            <div className="empty-state">
              <div className="empty-state-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="12" y1="8" x2="12" y2="12"></line>
                  <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
              </div>
              <h2>How can I support your clinical work today?</h2>
              <p>Ask about specific patient cases, evidence-based treatments, or make predictions based on clinical data.</p>
              <div className="sample-queries">
                <h3>Try these examples:</h3>
                <button onClick={() => setQuery("What are evidence-based approaches for treating treatment-resistant depression in adults?")}>
                  Evidence-based approaches for treatment-resistant depression
                </button>
                <button onClick={() => setQuery("Patient case: 28-year-old with panic disorder and agoraphobia, failed first-line SSRI treatment")}>
                  Patient with panic disorder not responding to first-line treatment
                </button>
                <button onClick={() => setQuery("Clinical prediction for 42-year-old female with recurring headaches, fatigue, and difficulty concentrating")}>
                  Clinical prediction for recurring headaches and cognitive symptoms
                </button>
              </div>
            </div>
          )}

          {loading && (
            <div className="loading">
              <div className="loading-spinner"></div>
              <p>Analyzing clinical query...</p>
            </div>
          )}

          {query && response && (
            <div className="chat-session">
              <div className="query-bubble">
                <p>{query}</p>
              </div>
              {renderResponse()}
            </div>
          )}
        </div>

        {searchHistory.length > 0 && (
          <aside className="search-history">
            <h3>Recent Queries</h3>
            <ul>
              {searchHistory.map((item, idx) => (
                <li key={idx} onClick={() => {
                  setQuery(item.query);
                  setResponse(item.response);
                  setQueryType(item.response.query_type);
                }}>
                  <span className="history-time">{item.timestamp}</span>
                  <span className="history-query">{item.query.substring(0, 30)}{item.query.length > 30 ? '...' : ''}</span>
                  <span className={`history-type ${item.response.query_type.toLowerCase()}`}>
                    {item.response.query_type === 'GENERAL' ? 'Clinical' : 
                     item.response.query_type === 'CASE' ? 'Case' : 
                     item.response.query_type === 'PREDICTION' ? 'Prediction' : 'Error'}
                  </span>
                </li>
              ))}
            </ul>
          </aside>
        )}
      </main>
    </div>
  );
}

export default App;