import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [recentSearches, setRecentSearches] = useState([]);
  const [selectedContext, setSelectedContext] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      
      const data = await response.json();
      setResults(data);
      
      // Add to recent searches
      if (!recentSearches.includes(query)) {
        const updatedSearches = [query, ...recentSearches.slice(0, 4)];
        setRecentSearches(updatedSearches);
        localStorage.setItem('recentSearches', JSON.stringify(updatedSearches));
      }
    } catch (error) {
      console.error('Error fetching results:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const savedSearches = localStorage.getItem('recentSearches');
    if (savedSearches) {
      setRecentSearches(JSON.parse(savedSearches));
    }
  }, []);

  // SVG Icons
  const SearchIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8"></circle>
      <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
    </svg>
  );

  const DocumentIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
      <polyline points="14 2 14 8 20 8"></polyline>
      <line x1="16" y1="13" x2="8" y2="13"></line>
      <line x1="16" y1="17" x2="8" y2="17"></line>
      <polyline points="10 9 9 9 8 9"></polyline>
    </svg>
  );

  const InfoIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"></circle>
      <line x1="12" y1="16" x2="12" y2="12"></line>
      <line x1="12" y1="8" x2="12.01" y2="8"></line>
    </svg>
  );

  return (
    <div className="legacysearch-container">
      <div className="legacysearch-header">
        <div className="legacysearch-logo">LegacySearch</div>
        <div className="legacysearch-tagline">Discover insights from archived conversations</div>
      </div>
      
      <div className="legacysearch-search-container">
        <form onSubmit={handleSearch} className="legacysearch-search-form">
          <div className="legacysearch-search-input-container">
            <span className="legacysearch-search-icon">
              <SearchIcon />
            </span>
            <input
              type="text"
              placeholder="Search for mental health topics, symptoms, or conditions (e.g., 'anxiety symptoms', 'coping strategies for depression')"
              className="legacysearch-search-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <button type="submit" className="legacysearch-search-button">
            Search
          </button>
        </form>
      </div>
      
      {!results && !loading && recentSearches.length > 0 && (
        <div className="legacysearch-recent-searches">
          <h3>Recent searches</h3>
          <ul className="legacysearch-search-list">
            {recentSearches.map((search, index) => (
              <li key={index} className="legacysearch-search-item">
                <button
                  onClick={() => {
                    setQuery(search);
                    handleSearch({ preventDefault: () => {} });
                  }}
                  className="legacysearch-search-item-button"
                >
                  <span className="legacysearch-search-item-icon"><SearchIcon /></span>
                  <span>{search}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
      
      {loading && (
        <div className="legacysearch-loader">
          <div className="legacysearch-spinner"></div>
          <div className="legacysearch-loader-text">Searching archives...</div>
        </div>
      )}
      
      {results && !loading && (
        <div className="legacysearch-results">
          <div className="legacysearch-results-header">
            <div className="legacysearch-results-icon">
              <DocumentIcon />
            </div>
            <h2 className="legacysearch-results-title">{results.title}</h2>
          </div>
          
          <div className="legacysearch-results-content">
            {results.contexts && results.contexts.length > 0 && (
              <div className="legacysearch-context-cards">
                {results.contexts.map((context, index) => (
                  <div key={index} className="legacysearch-context-card">
                    <div className="legacysearch-context-header">
                      <h4>Context {index + 1}</h4>
                      <button className="legacysearch-context-info-button" onClick={() => setSelectedContext(context)}>
                        <InfoIcon />
                        View Full Context
                      </button>
                    </div>
                    <p className="legacysearch-context-text">{context.substring(0, 150)}... <span className="legacysearch-context-link">Show more</span></p>
                    <div className="legacysearch-context-metadata">
                      <span>Source: {context.source}</span>
                      <span>Date: {context.date}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {selectedContext && (
              <div className="legacysearch-selected-context">
                <div className="legacysearch-context-header">
                  <h4>Selected Context</h4>
                  <button className="legacysearch-context-close-button" onClick={() => setSelectedContext(null)}>
                    &times;
                  </button>
                </div>
                <p className="legacysearch-context-text">{selectedContext}</p>
              </div>
            )}
            <div 
              className="legacysearch-response" 
              dangerouslySetInnerHTML={{ __html: results.response }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
