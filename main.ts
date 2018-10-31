const imdb = require('imdb-api')
const googleTrends = require('google-trends-api')
const fs = require('fs')
const path = require('path')

//container for  the module(to  be exported)
let lib = {};

// Base directory for the data folder
lib.baseDir = path.join(__dirname ,'/output');


console.log(lib.baseDir);

//write data to a file 
lib.create = function(file , data , callback ){
	//open the filefor writing
	fs.open(lib.baseDir + '/' + file + '.cvs', 'wx' , (err, fileDescriptor)=>{
		if(!err && fileDescriptor){
			//convert data to string
			let stringData = JSON.stringify(data);

			//write to file close.
			fs.writeFile(fileDescriptor, stringData , (err)=>{
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


//Delete a file
lib.delete = function(filename, callback){
	fs.unlink(lib.baseDir + '/' + filename + '.cvs', (err)=>{
		if(!err){
			callback(false);
		} else{
			callback('there was an error unlinking the files');
		}
	});

}

module.exports = lib;
