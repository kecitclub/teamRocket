const mongoose=require('mongoose')
 
const listingModel = require("../models/users.model")
const data=require("./users")

const db_config=require("../configs/db.config")


//connection with mongodb
main()
.then(()=>{
    console.log("Connected to Database")

}).catch((err)=>{
    console.log(err);
})

const initDB=async()=>{
    await listingModel.deleteMany({});
    await listingModel.insertMany(data.users)
    console.log("Data was initialized");
}

async function main(){
    await mongoose.connect(db_config.DB_URL)
}

initDB();