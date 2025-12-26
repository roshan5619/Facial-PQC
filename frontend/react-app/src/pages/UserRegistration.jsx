import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import BlinkCamera from '../components/BlinkCamera';
import { registrationAPI } from '../services/api';

const UserRegistration = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    full_name: '',
  });
  const [images, setImages] = useState({
    image1: null,
    image2: null,
    image3: null,
  });
  const [currentImageNumber, setCurrentImageNumber] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const [step, setStep] = useState(1); // 1: Form, 2: Images

  // ---------------------------------------
  // Handlers
  // ---------------------------------------

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleNextStep = (e) => {
    e.preventDefault();

    if (!formData.username || !formData.email || !formData.full_name) {
      setMessage({ type: 'error', text: 'Please fill all fields' });
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!emailRegex.test(formData.email)) {
      setMessage({ type: 'error', text: 'Please enter a valid email' });
      return;
    }

    if (formData.username.length < 3) {
      setMessage({ type: 'error', text: 'Username must be at least 3 characters' });
      return;
    }

    setMessage(null);
    setStep(2);
  };

  const openCamera = (imageNumber) => {
    setCurrentImageNumber(imageNumber);
    setShowCamera(true);
  };

  const handleCameraCapture = (capturedImage) => {
    const imageKey = `image${currentImageNumber}`;
    setImages({
      ...images,
      [imageKey]: capturedImage,
    });

    setShowCamera(false);
    setCurrentImageNumber(null);
  };

  const handleSubmit = async () => {
    if (!images.image1 || !images.image2 || !images.image3) {
      setMessage({ type: 'error', text: 'Please capture all 3 images' });
      return;
    }

    setLoading(true);
    setMessage(null);

    try {
      const response = await registrationAPI.register({
        ...formData,
        ...images,
      });

      setMessage({
        type: 'success',
        text: response.data.message || 'Registration successful! You can now login.',
      });

      setTimeout(() => navigate('/login'), 3000);
    } catch (error) {
      setMessage({
        type: 'error',
        text: error.response?.data?.detail || 'Registration failed. Please try again.',
      });
    } finally {
      setLoading(false);
    }
  };

  // ---------------------------------------

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 bg-gray-100">

      {/* Header */}
      <div className="text-center mb-10">
        <div className="text-6xl mb-2">✍️</div>
        <h1 className="text-3xl font-bold">Register New Account</h1>
        <p className="text-gray-600 mt-1">Create your account with face enrollment</p>
      </div>

      {/* Progress Indicator */}
      <div className="flex items-center justify-center mb-8 gap-6">

        {/* Step 1 indicator */}
        <div className="flex flex-col items-center">
          <div
            className={`w-10 h-10 flex items-center justify-center rounded-full border-2 ${
              step === 1
                ? 'bg-blue-600 border-blue-600 text-white'
                : 'border-gray-400 text-gray-400'
            }`}
          >
            1
          </div>
          <span className={step === 1 ? 'text-blue-600' : 'text-gray-400'}>
            Details
          </span>
        </div>

        {/* Connector */}
        <div className={`h-1 w-10 ${step === 2 ? 'bg-blue-600' : 'bg-gray-300'}`}></div>

        {/* Step 2 indicator */}
        <div className="flex flex-col items-center">
          <div
            className={`w-10 h-10 flex items-center justify-center rounded-full border-2 ${
              step === 2
                ? 'bg-blue-600 border-blue-600 text-white'
                : 'border-gray-400 text-gray-400'
            }`}
          >
            2
          </div>
          <span className={step === 2 ? 'text-blue-600' : 'text-gray-400'}>
            Face Enrollment
          </span>
        </div>
      </div>

      {/* Message */}
      {message && (
        <div
          className={`mb-6 p-4 rounded-lg w-full max-w-lg ${
            message.type === 'success'
              ? 'bg-green-100 text-green-800 border border-green-300'
              : 'bg-red-100 text-red-800 border border-red-300'
          }`}
        >
          {message.text}
        </div>
      )}

      {/* Step 1: User Details Form */}
      {step === 1 && (
        <div className="w-full max-w-lg bg-white p-6 rounded-xl shadow">

          <div className="mb-4">
            <label className="font-semibold">Username *</label>
            <input
              type="text"
              name="username"
              value={formData.username}
              onChange={handleInputChange}
              className="w-full p-2 mt-1 border rounded-lg"
            />
          </div>

          <div className="mb-4">
            <label className="font-semibold">Email *</label>
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleInputChange}
              className="w-full p-2 mt-1 border rounded-lg"
            />
          </div>

          <div className="mb-6">
            <label className="font-semibold">Full Name *</label>
            <input
              type="text"
              name="full_name"
              value={formData.full_name}
              onChange={handleInputChange}
              className="w-full p-2 mt-1 border rounded-lg"
            />
          </div>

          <button
            onClick={handleNextStep}
            className="w-full bg-blue-600 text-white p-3 rounded-lg font-semibold hover:bg-blue-700"
          >
            Next: Capture Face Images →
          </button>

        </div>
      )}

      {/* Step 2: Face Enrollment */}
      {step === 2 && (
        <div className="w-full max-w-lg bg-white p-6 rounded-xl shadow">

          <p className="text-gray-600 mb-4">
            Important: Capture <b>3 images</b> of your face from different angles.
          </p>

          <div className="grid grid-cols-1 gap-6">

            {[1, 2, 3].map((num) => (
              <div key={num} className="border p-4 rounded-lg flex flex-col gap-3">

                {/* Image Preview */}
                <div className="h-40 bg-gray-100 rounded-lg flex items-center justify-center">
                  {images[`image${num}`] ? (
                    <span className="text-green-600 font-semibold">Captured ✓</span>
                  ) : (
                    <span className="text-gray-400">No image</span>
                  )}
                </div>

                {/* Capture Button */}
                <button
                  type="button"
                  onClick={() => openCamera(num)}
                  className="w-full bg-blue-600 text-white px-3 py-2 rounded-lg text-sm hover:bg-blue-700"
                >
                  {images[`image${num}`] ? 'Retake' : 'Capture'} #{num}
                </button>

                {/* Label */}
                <p className="text-sm text-gray-500">
                  {num === 1 && 'Front view'}
                  {num === 2 && 'Left angle'}
                  {num === 3 && 'Right angle'}
                </p>

              </div>
            ))}
          </div>

          <div className="flex mt-6 gap-4">

            <button
              onClick={() => setStep(1)}
              className="flex-1 p-3 border-2 border-gray-300 rounded-lg font-semibold hover:bg-gray-50"
            >
              ← Back
            </button>

            <button
              onClick={handleSubmit}
              disabled={loading}
              className="flex-1 p-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 disabled:opacity-50"
            >
              {loading ? 'Registering...' : 'Complete Registration ✓'}
            </button>
          </div>
        </div>
      )}

      {/* Back to Home */}
      <button
        onClick={() => navigate('/')}
        className="text-blue-600 hover:text-blue-800 text-sm font-medium mt-6"
      >
        ← Back to Home
      </button>

      {/* BlinkCamera Modal */}
      {showCamera && (
        <BlinkCamera
          onCapture={handleCameraCapture}
          onClose={() => {
            setShowCamera(false);
            setCurrentImageNumber(null);
          }}
          requireBlink={true}
          autoCapture={true}
        />
      )}
    </div>
  );
};

export default UserRegistration;
