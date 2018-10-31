const imdb = require('imdb-api')
const googleTrends = require('google-trends-api')
const fs = require('fs')
const path = require('path')

//container for  the module(to  be exported)
let lib = {};

// Base directory for the data folder
lib.baseDir = path.join(__dirname ,'/output');

//handler for errors
let handler = function(err){
	console.log(err);
}

console.log(lib.baseDir);
//const a = lib.read('MovieData');
//lib.delete("test", handler);
//lib.create("data" , "test", handler);


//write data to a file 
lib.create = function(file , data , callback ){
	//open the filefor writing
	fs.open(lib.baseDir + '/' + file + '.csv', 'wx' , (err, fileDescriptor)=>{
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
				}-+
			});
		} else {
			callback('Could not create new file, it may already exist');
		}
	});
};


//Read data from a file 
lib.read = function(file){
	let data = fs.readFileSync(__dirname+'/'+file+'.csv' , 'utf-8' );
	return data;
}

//Delete a file
lib.delete = function(filename, callback){
	fs.unlinkSync(lib.baseDir + '/' + filename + '.csv', (err)=>{
		if(!err){
			callback(false);
		} else{
			callback('there was an error unlinking the files');
		}
	});

}

//adds modularity for possible future use
module.exports = lib;
