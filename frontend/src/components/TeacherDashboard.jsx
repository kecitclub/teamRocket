import React from 'react';

const TeacherDashboard = () => {
  return (
    <div className="min-h-screen bg-blue-50 flex flex-col items-center justify-center">
      <div className="bg-white shadow-md rounded-lg p-8 w-full max-w-2xl">
        <h1 className="text-3xl font-bold text-blue-800 text-center mb-6">Teacher Dashboard</h1>
        <p className="text-gray-700 text-lg text-center">
          Welcome, Teacher! Here you can manage your courses, view student performance, and access teaching materials.
        </p>
        <div className="mt-8 flex justify-center space-x-4">
          <button className="rounded-md bg-blue-600 px-4 py-2 text-white font-semibold hover:bg-blue-500">
            Manage Courses
          </button>
          <button className="rounded-md bg-blue-600 px-4 py-2 text-white font-semibold hover:bg-blue-500">
            View Students
          </button>
        </div>
      </div>
    </div>
  );
};

export default TeacherDashboard;
