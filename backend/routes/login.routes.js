const loginController=require("../controllers/login.controller")
module.exports=(app) =>{
    app.get("/",loginController.index);
    app.post("/login",loginController.login);

}