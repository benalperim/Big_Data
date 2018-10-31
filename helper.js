
const fs = require('fs')
const path = require('path')

//container for  the module(to  be exported)
let helper = {};

// Base directory for the data folder
helper.baseDir = path.join(__dirname ,'/output');

//handler for errors
let handler = function(err){
	console.log(err);
}

console.log(helper.baseDir);

//helper.delete("test", handler);
//helper.create("data" , "test", handler);


//write data to a file 
helper.create = function(file , data , callback ){
	//open the filefor writing
	fs.open(helper.baseDir + '/' + file + '.csv', 'wx' , (err, fileDescriptor)=>{
		if(!err && fileDescriptor){
			//convert data to string FIX HERE FROM API CALL RETURNS A JSON (JSON.parse())
			let stringData = data;

			//write to file close.
			fs.writeFileSync(fileDescriptor, stringData , (err)=>{
				if(!err){
					fs.close(fileDescriptor , (err)=>{
						if(!err){
							callback(false);
						} else {
							callback('Eror closing the file');
						}
					});
				} else{
					callback('Error writing the file');
				}
			});
		} else {
			callback('Could not create new file, it may already exist');
		}
	});
};


//Read data from a file 
helper.read = function(file){
	let data = fs.readFileSync(__dirname+'/'+file+'.csv' , 'utf-8' );
	return data;
}


//Delete a file
helper.delete = function(filename, callback){
	fs.unlinkSync(helper.baseDir + '/' + filename + '.csv', (err)=>{
		if(!err){
			callback(false);
		} else{
			callback('there was an error unlinking the files');
		}
	});
}



//adds modularity for possible future use
module.exports = helper;
