const googleTrends = require('google-trends-api')
const helper = require('./helper')

let readData = [];
let outputData = '';
let TestArray = []
readData = helper.read('GoogleTrendSearch').split('\n')
console.log(readData.length);//Should come out as  1567 and it does





const trendData = async function(readData , outputData){

    for(let  i =0; i < readData.length; i++){



        let keyArray = readData[i].split(',' , 5)
        let Data  =readData[i].split(',')
        let startTime = new Date(Data[5].split('-')[0])
        let endTime = new Date(Data[5].split('-')[1])
        let Identifier = Data[6]
        //console.log(keyArray)
        //console.log(startTime)
        let payload = await googleTrends
            .interestOverTime({
                keyword: keyArray,
                startTime,
                endTime,
                geo: 'US',
                //granularTimeResolution: true
            }).then((payload) => {
                //outputData += payload.split("averages\":")[1] + "\n"

                console.log(payload.split("averages\":")[1] + ", " + Identifier)
                
            }).catch((err) => {
                console.log(err);
            });
    }
}(readData , outputData)

let callbak = function(err){
    if (err)
     console.log(err) 
 } 
// let test1 = "muahahahah\n"readData
 helper.update('TrendsOutput',  outputData, callbak )
//  let test2 = "hoep this freaking works "
//  helper.update('TrendsOutput',  test2,callbak )