import React from 'react';

const StudentDashboard = () => {
  return (
    <div className="min-h-screen bg-green-50 flex flex-col items-center justify-center">
      <div className="bg-white shadow-md rounded-lg p-8 w-full max-w-2xl">
        <h1 className="text-3xl font-bold text-green-800 text-center mb-6">Student Dashboard</h1>
        <p className="text-gray-700 text-lg text-center">
          Welcome, Student! Here you can view your courses, assignments, and other resources.
        </p>
        <div className="mt-8 flex justify-center">
          <button className="rounded-md bg-green-600 px-4 py-2 text-white font-semibold hover:bg-green-500">
            View Courses
          </button>
        </div>
      </div>
    </div>
  );
};

export default StudentDashboard;
