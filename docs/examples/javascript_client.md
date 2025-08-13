# JavaScript/React Client Examples

This guide provides examples for integrating with the EduPulse Analytics API using JavaScript and React.

## Installation

```bash
npm install axios react-query recharts
# or
yarn add axios react-query recharts
```

## Basic API Client

### Axios Configuration

```javascript
// api/client.js
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle auth errors
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // Redirect to login or refresh token
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default apiClient;
```

## React Hooks

### usePrediction Hook

```javascript
// hooks/usePrediction.js
import { useState, useCallback } from 'react';
import { useQuery, useMutation } from 'react-query';
import apiClient from '../api/client';

export const usePrediction = (studentId) => {
  const { data, isLoading, error, refetch } = useQuery(
    ['prediction', studentId],
    async () => {
      const response = await apiClient.post('/api/v1/predictions/predict', {
        student_id: studentId,
        include_factors: true,
      });
      return response.data;
    },
    {
      enabled: !!studentId,
      staleTime: 5 * 60 * 1000, // 5 minutes
    }
  );

  return {
    prediction: data,
    isLoading,
    error,
    refetch,
  };
};

// Batch predictions hook
export const useBatchPredictions = () => {
  const mutation = useMutation(
    async (studentIds) => {
      const response = await apiClient.post('/api/v1/predictions/predict-batch', {
        student_ids: studentIds,
        top_k: studentIds.length,
      });
      return response.data;
    }
  );

  return {
    predictBatch: mutation.mutate,
    isLoading: mutation.isLoading,
    data: mutation.data,
    error: mutation.error,
  };
};
```

## React Components

### Risk Dashboard Component

```jsx
// components/RiskDashboard.jsx
import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, PieChart, Pie, Cell
} from 'recharts';
import { useBatchPredictions } from '../hooks/usePrediction';

const RISK_COLORS = {
  low: '#4ade80',
  medium: '#fbbf24',
  high: '#fb923c',
  critical: '#ef4444',
};

export const RiskDashboard = ({ studentIds }) => {
  const { predictBatch, data, isLoading } = useBatchPredictions();
  const [statistics, setStatistics] = useState(null);

  useEffect(() => {
    if (studentIds?.length > 0) {
      predictBatch(studentIds);
    }
  }, [studentIds]);

  useEffect(() => {
    if (data?.predictions) {
      // Calculate statistics
      const stats = {
        total: data.predictions.length,
        byCategory: {},
        avgRiskScore: 0,
      };

      data.predictions.forEach((pred) => {
        stats.byCategory[pred.risk_category] = 
          (stats.byCategory[pred.risk_category] || 0) + 1;
        stats.avgRiskScore += pred.risk_score;
      });

      stats.avgRiskScore /= stats.total;
      setStatistics(stats);
    }
  }, [data]);

  if (isLoading) return <div className="loading">Loading predictions...</div>;
  if (!statistics) return null;

  const pieData = Object.entries(statistics.byCategory).map(([category, count]) => ({
    name: category,
    value: count,
    color: RISK_COLORS[category],
  }));

  return (
    <div className="risk-dashboard">
      <h2>Risk Assessment Dashboard</h2>
      
      <div className="stats-grid">
        <div className="stat-card">
          <h3>Total Students</h3>
          <p className="stat-value">{statistics.total}</p>
        </div>
        
        <div className="stat-card">
          <h3>Average Risk Score</h3>
          <p className="stat-value">{(statistics.avgRiskScore * 100).toFixed(1)}%</p>
        </div>
        
        <div className="stat-card">
          <h3>High Risk Students</h3>
          <p className="stat-value">
            {(statistics.byCategory.high || 0) + (statistics.byCategory.critical || 0)}
          </p>
        </div>
      </div>

      <div className="charts-container">
        <div className="chart">
          <h3>Risk Distribution</h3>
          <PieChart width={400} height={300}>
            <Pie
              data={pieData}
              cx={200}
              cy={150}
              outerRadius={100}
              fill="#8884d8"
              dataKey="value"
              label
            >
              {pieData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </div>

        <div className="chart">
          <h3>Students by Risk Category</h3>
          <BarChart width={400} height={300} data={pieData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#8884d8">
              {pieData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </div>
      </div>
    </div>
  );
};
```

### Student Risk Card

```jsx
// components/StudentRiskCard.jsx
import React from 'react';
import { usePrediction } from '../hooks/usePrediction';

export const StudentRiskCard = ({ studentId, studentInfo }) => {
  const { prediction, isLoading, error } = usePrediction(studentId);

  if (isLoading) return <div className="card loading">Loading...</div>;
  if (error) return <div className="card error">Error loading prediction</div>;
  if (!prediction) return null;

  const getRiskColor = (category) => {
    const colors = {
      low: 'green',
      medium: 'yellow',
      high: 'orange',
      critical: 'red',
    };
    return colors[category] || 'gray';
  };

  return (
    <div className={`student-card risk-${prediction.risk_category}`}>
      <div className="card-header">
        <h3>{studentInfo.first_name} {studentInfo.last_name}</h3>
        <span className="student-id">{studentId}</span>
      </div>

      <div className="risk-indicator" style={{ 
        backgroundColor: getRiskColor(prediction.risk_category) 
      }}>
        <span className="risk-score">{(prediction.risk_score * 100).toFixed(0)}%</span>
        <span className="risk-category">{prediction.risk_category}</span>
      </div>

      <div className="contributing-factors">
        <h4>Top Risk Factors:</h4>
        <ul>
          {prediction.contributing_factors?.slice(0, 3).map((factor, idx) => (
            <li key={idx}>
              <strong>{factor.factor}:</strong> {factor.details}
            </li>
          ))}
        </ul>
      </div>

      <div className="card-actions">
        <button className="btn btn-primary">View Details</button>
        <button className="btn btn-secondary">Create Intervention</button>
      </div>
    </div>
  );
};
```

## Real-time Updates

### WebSocket Connection

```javascript
// api/websocket.js
class PredictionWebSocket {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.listeners = new Map();
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.authenticate();
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleMessage(data);
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      // Reconnect after 5 seconds
      setTimeout(() => this.connect(), 5000);
    };
  }

  authenticate() {
    const token = localStorage.getItem('auth_token');
    this.send({
      type: 'auth',
      token: token,
    });
  }

  subscribe(studentId, callback) {
    this.listeners.set(studentId, callback);
    this.send({
      type: 'subscribe',
      student_id: studentId,
    });
  }

  unsubscribe(studentId) {
    this.listeners.delete(studentId);
    this.send({
      type: 'unsubscribe',
      student_id: studentId,
    });
  }

  handleMessage(data) {
    if (data.type === 'prediction_update') {
      const callback = this.listeners.get(data.student_id);
      if (callback) {
        callback(data.prediction);
      }
    }
  }

  send(data) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

// Usage in React component
const useRealtimePrediction = (studentId) => {
  const [prediction, setPrediction] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    wsRef.current = new PredictionWebSocket('ws://localhost:8000/ws');
    wsRef.current.connect();

    wsRef.current.subscribe(studentId, (newPrediction) => {
      setPrediction(newPrediction);
    });

    return () => {
      wsRef.current.unsubscribe(studentId);
      wsRef.current.disconnect();
    };
  }, [studentId]);

  return prediction;
};
```

## Form Components

### Prediction Request Form

```jsx
// components/PredictionForm.jsx
import React, { useState } from 'react';
import { useMutation } from 'react-query';
import apiClient from '../api/client';

export const PredictionForm = ({ onPredictionComplete }) => {
  const [formData, setFormData] = useState({
    student_id: '',
    reference_date: new Date().toISOString().split('T')[0],
    include_factors: true,
  });

  const mutation = useMutation(
    async (data) => {
      const response = await apiClient.post('/api/v1/predictions/predict', data);
      return response.data;
    },
    {
      onSuccess: (data) => {
        onPredictionComplete(data);
      },
    }
  );

  const handleSubmit = (e) => {
    e.preventDefault();
    mutation.mutate(formData);
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  return (
    <form onSubmit={handleSubmit} className="prediction-form">
      <div className="form-group">
        <label htmlFor="student_id">Student ID:</label>
        <input
          type="text"
          id="student_id"
          name="student_id"
          value={formData.student_id}
          onChange={handleChange}
          required
        />
      </div>

      <div className="form-group">
        <label htmlFor="reference_date">Reference Date:</label>
        <input
          type="date"
          id="reference_date"
          name="reference_date"
          value={formData.reference_date}
          onChange={handleChange}
        />
      </div>

      <div className="form-group">
        <label>
          <input
            type="checkbox"
            name="include_factors"
            checked={formData.include_factors}
            onChange={handleChange}
          />
          Include Contributing Factors
        </label>
      </div>

      <button 
        type="submit" 
        disabled={mutation.isLoading}
        className="btn btn-primary"
      >
        {mutation.isLoading ? 'Predicting...' : 'Get Prediction'}
      </button>

      {mutation.error && (
        <div className="error-message">
          Error: {mutation.error.response?.data?.detail || 'Something went wrong'}
        </div>
      )}
    </form>
  );
};
```

## Error Handling

### Error Boundary Component

```jsx
// components/ErrorBoundary.jsx
import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
    
    // Send error to monitoring service
    if (window.errorReporter) {
      window.errorReporter.captureException(error, {
        extra: errorInfo,
      });
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h2>Something went wrong</h2>
          <details style={{ whiteSpace: 'pre-wrap' }}>
            {this.state.error?.toString()}
          </details>
          <button onClick={() => window.location.reload()}>
            Reload Page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
```

## Testing

### Component Tests

```javascript
// __tests__/StudentRiskCard.test.js
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { StudentRiskCard } from '../components/StudentRiskCard';
import { rest } from 'msw';
import { setupServer } from 'msw/node';

const server = setupServer(
  rest.post('/api/v1/predictions/predict', (req, res, ctx) => {
    return res(
      ctx.json({
        student_id: 'STU001',
        risk_score: 0.75,
        risk_category: 'high',
        contributing_factors: [
          { factor: 'attendance', details: 'High absence rate' },
        ],
      })
    );
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

test('renders student risk card with prediction', async () => {
  const queryClient = new QueryClient();
  
  render(
    <QueryClientProvider client={queryClient}>
      <StudentRiskCard 
        studentId="STU001" 
        studentInfo={{ first_name: 'John', last_name: 'Doe' }}
      />
    </QueryClientProvider>
  );

  // Check loading state
  expect(screen.getByText(/Loading.../i)).toBeInTheDocument();

  // Wait for prediction to load
  await waitFor(() => {
    expect(screen.getByText('John Doe')).toBeInTheDocument();
    expect(screen.getByText('75%')).toBeInTheDocument();
    expect(screen.getByText('high')).toBeInTheDocument();
  });
});
```

## Next Steps

- See [Python Client Examples](./python_client.md) for backend integration
- Review [Data Import Examples](./data_import.md) for bulk data loading
- Check the [API Reference](../API_REFERENCE.md) for complete endpoint documentation