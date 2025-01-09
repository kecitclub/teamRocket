import { useState } from 'react'
import './App.css'
import { Route,Routes } from 'react-router-dom'
import Home from './components/Home'
import Hero from './components/Hero'
import StudentDashboard from './components/StudentDashboard'
import TeacherDashboard from './components/TeacherDashboard'
import AdminDashboard from './components/AdminDashboard'
function App() {


  return (
    <>
      <div className='bg-red-600'>
        <Routes>
          <Route path='/' element={ < Hero />} />
          <Route path='/Home' element={ < Home />} />
          <Route path="/student-dashboard" element={<StudentDashboard />} />
          <Route path="/teacher-dashboard" element={<TeacherDashboard />} />
          <Route path="/admin-dashboard" element={<AdminDashboard />} />
        </Routes>
      </div>
    </>
  )
}

export default App
