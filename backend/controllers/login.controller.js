//users model
const listingModel = require("../models/users.model")
//index route
exports.index =async (req, res) =>{
    try {
        res.send("Login Page");
    } catch (error) {
        console.log(error);
    }
}



exports.login = async (req, res) => {
  const { email, password } = req.body;
    console.log(email);
    console.log(password);
  try {
    const user = await listingModel.findOne({ email, password });

    if (!user) {
      console.log("User not found");
      return res.status(401).json({ message: "Invalid email or password" });
    }
    console.log("User found");
    console.log(user.role);
    // Respond with user role
    return res.status(200).json({ message: "Login successful", role: user.role });
  } catch (error) {
    console.error("Error during login:", error);
    return res.status(500).json({ message: "Server error" });
  }
};
