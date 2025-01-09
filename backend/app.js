//express 
const express = require('express')
const app = express()
const path = require('path');
const cors = require('cors');

//mongodb
const mongoose = require("mongoose")

app.use(cors());
//configs
const server_config=require("./configs/server.config")
const db_config=require("./configs/db.config")

// Middleware to parse JSON bodies
app.use(express.json());

//connection with mongodb
main()
.then(()=>{
    
    console.log("Connected to Database")

}).catch((err)=>{
    console.log(err);
})


async function main(){
    mongoose.set('strictQuery', true);
    await mongoose.connect(db_config.DB_URL)
}

//stiching homeroute
require("./routes/login.routes")(app)

//connection with server
app.listen(server_config.PORT,()=>{
    console.log("Server is listening to the port ",server_config.PORT)
})