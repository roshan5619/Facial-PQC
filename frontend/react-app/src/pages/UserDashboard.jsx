import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { authAPI } from '../services/api';

const UserDashboard = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [sessionTimer, setSessionTimer] = useState(300); // 5 minutes in seconds

// eslint-disable-next-line react-hooks/exhaustive-deps
useEffect(() => {
  verifyUser();
    const interval = setInterval(() => {
      setSessionTimer((prev) => {
        if (prev <= 1) {
          handleLogout();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
}, []);


  const verifyUser = async () => {
    try {
      const response = await authAPI.verifyToken();
      setUser(response.data);
    } catch (error) {
      console.error('Verification failed:', error);
      navigate('/');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await authAPI.logout();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      localStorage.removeItem('access_token');
      localStorage.removeItem('username');
      navigate('/');
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-xl font-semibold">
        Loading...
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6 flex flex-col items-center bg-gray-100 gap-8">

      {/* Success Banner */}
      <div className="bg-green-100 border border-green-300 p-6 rounded-2xl shadow w-full max-w-2xl">
        <div className="flex items-center gap-3 text-green-800">
          <div className="text-4xl">✓</div>
          <div>
            <h2 className="font-bold text-xl">Login Successful!</h2>
            <p>Welcome back, {user?.username}</p>
          </div>
        </div>

        <div className="flex justify-between mt-4 text-sm text-green-700">
          <span>Session expires in {formatTime(sessionTimer)}</span>

          {sessionTimer <= 60 && (
            <span className="text-red-600 font-semibold flex items-center gap-1">
              ⚠️ Session expiring soon
            </span>
          )}
        </div>
      </div>

      {/* User Info Card */}
      <div className="bg-white p-6 rounded-2xl shadow-xl w-full max-w-2xl">
        <h3 className="text-xl font-bold mb-4">Account Information</h3>

        <div className="space-y-3 text-gray-700">
          <p><strong>User ID:</strong> {user?.user_id}</p>
          <p><strong>Username:</strong> {user?.username}</p>
          <p><strong>Email:</strong> {user?.email}</p>
          <p><strong>Full Name:</strong> {user?.full_name}</p>
        </div>
      </div>

      {/* Authentication Method */}
      <div className="bg-white p-6 rounded-2xl shadow-xl w-full max-w-2xl">
        <h3 className="text-xl font-bold mb-4">Authentication Method</h3>

        <div className="flex items-center gap-4">
          <div className="text-5xl">🤖</div>

          <div>
            <p className="font-semibold">AI-Powered Face Recognition</p>
            <p className="text-gray-600 text-sm">
              Verified with liveness detection & quantum ML
            </p>
          </div>

          <div className="ml-auto bg-green-600 text-white px-4 py-2 rounded-lg shadow text-sm">
            Verified & Secure
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4 w-full max-w-2xl">
        <button
          onClick={handleLogout}
          className="flex-1 bg-red-600 text-white px-6 py-4 rounded-xl font-semibold hover:bg-red-700 transition-colors shadow-lg"
        >
          Logout
        </button>

        <button
          onClick={() => navigate('/')}
          className="flex-1 bg-gray-600 text-white px-6 py-4 rounded-xl font-semibold hover:bg-gray-700 transition-colors shadow-lg"
        >
          Back to Home
        </button>
      </div>

      {/* Footer Info */}
      <p className="text-gray-600 text-sm text-center">
        You will be automatically logged out after {formatTime(sessionTimer)} of session time.
      </p>
    </div>
  );
};

export default UserDashboard;
