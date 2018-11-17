const imdb = require('imdb-api')
const googleTrends = require('google-trends-api')
const helper = require('./helper')

let apiKey = '6d65d3e'
let MovieData = [];
let outputData = '';

//Create api key for multiple calls 
const cli = new imdb.Client({apiKey});

//cli.get({'name': 'The Toxic Avenger'}).then((search) => {
	
//	  console.log(search);
//	})

const handler = function(err){
	console.log(err);
}


//Get the movie data and parse each entry as a element in the array
MovieData = helper.read('MovieData').split('\n')

console.log(MovieData.length)//Should come out as 315 and it does

const apiCalls =async function(MovieData , outputData, callback){
	
	// do async calls for multiple data retrivals
	for(let  i = 0; i < MovieData.length; i++){
		try{
		//get the api call back
		let payload =  await cli.get({'name': MovieData[i]})
		
		//get the actor director prod and genre fields from payload
		outputData = payload.actors + " (actors) " + payload.director + " (directors) " + payload.production + " (Prod) " + payload.genres + " (Genres ) \n"
		console.log(outputData)
		}catch(err) {
			console.log(MovieData[i] + " could not return anything")
			//callback("bad movie")
		}
	}
	return outputData

}(MovieData, outputData, handler)
console.log(outputData)

helper.create('output' , outputData , handler)