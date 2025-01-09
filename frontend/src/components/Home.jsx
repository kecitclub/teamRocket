
  import React, { useState } from 'react';
  import { useNavigate } from 'react-router-dom';
  import axios from 'axios';
  
  const Home = () => {
    const [userData, setUserData] = useState({
      email: '',
      password: '',
    });
    const navigate = useNavigate();
  
    const handleInputChange = (e) => {
      const { name, value } = e.target;
      setUserData((prevData) => ({
        ...prevData,
        [name]: value,
      }));
    };
  
    const handleSubmit = async (e) => {
        e.preventDefault();
        // console.log('User Data:', userData);
      
        try {
          const response = await axios.post(`http://localhost:8080/login`, userData);
            
          if (response.status === 200) {
            const { role } = response.data;
            // Redirect based on role
            if (role === 'student') {
              console.log('student');
              navigate('/student-dashboard');
            } else if (role === 'teacher') {
              console.log('teacher');
              navigate('/teacher-dashboard');
            } else if (role === 'admin') { // Removed extra space
              console.log('admin');
              navigate('/admin-dashboard');
            }
          }
        } catch (error) {
          console.error('Login error:', error.response?.data || error.message);
          alert('Invalid credentials or server error.');
        }
      };
      
  
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="bg-white shadow-md rounded-lg p-8 w-full max-w-sm">
          <h2 className="text-2xl font-bold text-gray-800 text-center mb-6">Login</h2>
          <form className="space-y-4" onSubmit={handleSubmit}>
            {/* Email */}
            <div>
              <label
                htmlFor="email"
                className="block text-sm font-medium text-gray-700"
              >
                Email Address
              </label>
              <input
                type="email"
                id="email"
                name="email"
                value={userData.email}
                onChange={handleInputChange}
                required
                className="mt-2 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                placeholder="Enter your email"
              />
            </div>
  
            {/* Password */}
            <div>
              <label
                htmlFor="password"
                className="block text-sm font-medium text-gray-700"
              >
                Password
              </label>
              <input
                type="password"
                id="password"
                name="password"
                value={userData.password}
                onChange={handleInputChange}
                required
                className="mt-2 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                placeholder="Enter your password"
              />
            </div>
  
            {/* Submit Button */}
            <div>
              <button
                type="submit"
                className="w-full rounded-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white hover:bg-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
              >
                Login
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  };
  


export default Home;