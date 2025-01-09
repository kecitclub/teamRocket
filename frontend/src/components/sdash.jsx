import React from "react";
import "./sdash.css"; // Make sure the CSS file is linked

const Dashboard = () => {
  return (
    <div id="background">
      {/* Left Sidebar */}
      <div id="leftbg">
        <div id="logo">
          <img src="logow.svg" alt="Logo" />
          <h2>LEARNOVA</h2>
        </div>
        <div id="overview">
          <h2>OVERVIEW</h2>
          <div className="ovList">
            <a href="sdashboard.html">Dashboard</a>
            <a href="sdashboard.html">Exams</a>
            <a href="sdashboard.html">Resources</a>
          </div>
        </div>
        <div id="final">
          <div className="settings">
            <img src="settingsw.png" alt="Settings" />
            <h2>Settings</h2>
          </div>
          <div className="logout">
            <img src="logout.png" alt="Log Out" />
            <h2>Log Out</h2>
          </div>
        </div>
      </div>

      {/* Middle Content */}
      <div id="middlebg">
        <form action="/search" method="GET">
          <input type="search" name="q" placeholder="Search..." required />
          <button type="submit">
            <img src="search.png" alt="Search" />
          </button>
        </form>
      </div>

      {/* Right Sidebar */}
      <div id="rightbg"></div>
    </div>
  );
};

export default Dashboard;
