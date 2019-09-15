import Spinner from 'react-spinner-material';
import React from 'react';
import axios from 'axios';
import { renderRoutes } from 'react-router-config';
import { Breadcrumb } from './components';
import { rootPath } from './routes';
//Logo of Home page
import ZetaGAN_logo from './images/ZetaGAN_logo.png'
//CSV support
import { CsvToHtmlTable } from "react-csv-to-table";
import { CSVLink } from "react-csv";
//animation files
import panda1 from './images/panda1.gif'
import tiger1 from './images/tiger.gif'
import get_started from './images/get_started.gif'
//static sample data
import { sampleData_gan } from "./samples/sample_gan";
import { sampleData_smote } from "./samples/sample_smote";
import { perf } from "./perf";
  
  
/**
 * These are root pages
 */
const Home = ({ location }) => {
  return (
    <div>
      <h1 className="py-3">Home</h1>
      <Breadcrumb locationPath={location.pathname} />
      <img src={ZetaGAN_logo} alt="team logo" />
    </div>
  );
};


/**
 * These are 2nd pages - Generate Data
 */
const Books = ({ location }) => {
    const onMatchedRoutes = (matchedRoutes) => {
      return [
        {
          route: {
            path: `${rootPath}/`,
            breadcrumbName: 'Home'
          }
        },
        ...matchedRoutes
      ];
    };

    class FormBox extends React.Component{

      // Constructor
      constructor() {
        super()
        this.state = {
          previewImageUrl: false,
          imageHeight: 200,
          imagePrediction: "",
          num_of_data : '2000',
          csvData: '',
          isVisibleAll: false,
          isHidden: true,
          isHidden2: true,
          isVisible: false,
          isVisible2: false,
          result_of_smote: [],
          test2: [],
        }
        this.getInitialState = this.getInitialState.bind(this)
        this.pressButton = this.pressButton.bind(this)
        this.getPhoto = this.getPhoto.bind(this)
        this.changeNumOfDataState = this.changeNumOfDataState.bind(this)
      }


      getInitialState() {
        return {
          file: '',
          imagePreviewUrl: ''
        }
      }

      pressButton(e) {
        e.preventDefault();
        // TODO: do something with -> this.state.file

        var self = this;
        const formData = new FormData()
        formData.append('file', this.state.imageFile, 'img.png')

        this.setState({isVisibleAll: true})
        this.setState({isVisible: true})
        this.setState({isVisible2: true})
        this.setState({isHidden: true})
        this.setState({isHidden2: true})
    
        var t0 = performance.now();
        axios.defaults.timeout = 20000;
        axios({url:'http://35.229.179.160:5000/gan', method:'get', timeout:20000})
        .then(function(response, data) {
            data = response.data;
            self.setState({test2:data})
            self.setState({isVisible:false})
            self.setState({isHidden: false})
            var t1 = performance.now();
            console.log("The time it took to generate data by WGAN " + (t1 - t0) + " milliseconds.")
        })

        console.log('handle uploading-WGAN', this.state.file);

        axios({url:'http://35.229.179.160:5000/smote', method:'get', timeout:20000})
        .then(function(response, data) {
            data = response.data;
            self.setState({result_of_smote:data})
            self.setState({isVisible2:false})
            self.setState({isHidden2: false})
            var t1 = performance.now();
            console.log("The time it took to generate data by SMOTE " + (t1 - t0) + " milliseconds.")
            console.log(response);
        })
        .catch(function (error) {
            console.log('Error:',error.message);
        })

        console.log('handle uploading-SMOTE', this.state.file);
      }

      getPhoto(e) {
        e.preventDefault();

        let reader = new FileReader();
        let file = e.target.files[0];

        // If the image upload is cancelled
        if (!file) {
          return
        }

        this.setState({imageFile: file})

        reader.onloadend = () => {
          this.setState({
            file: file,
            imagePreviewUrl: reader.result
          });
        }

        reader.readAsDataURL(file);
      }

      changeNumOfDataState(event) {
        this.setState({num_of_data:event.target.value})
      }

      /* render contents  */
      render() {
        let {imagePreviewUrl} = this.state;
        let imagePreview = null;
        if (imagePreviewUrl) {
          imagePreview = (<img alt="..." src={imagePreviewUrl} />);
        } else {
          imagePreview = (<div className="previewText">Please select a dataset.</div>);
        }


        const ShowWGANResult = () =>
        <div>
        {
          Object.entries(this.state.test2)
          .map( ([key, value]) => <div><i>${key} is {value}</i><br/></div> )
        }
        </div>

        const ShowSMOTEResult = () =>
        <div>
        {
          Object.entries(this.state.result_of_smote)
          .map( ([key, value]) => <div><i>${key} is ${value}</i><br/></div> )
        }
        </div>

        return (
          <div>
            <h1 className="py-3">Generate Data</h1>
            <Breadcrumb
              locationPath={location.pathname}
              onMatchedRoutes={onMatchedRoutes}
            />

            <h3>Input original data</h3>
            <form action='.' enctype="multipart/form-data">
              <label>Number of Data:</label>
              <input type="text" id="num_of_data" name="num_of_data" value={this.state.num_of_data} onChange={this.changeNumOfDataState} />
              <br/>
              {/* Button for choosing an image */}
              <input type='file' name="file" onChange={this.getPhoto}/>
              {/* Button for sending image to backend */}
              <button type="submit" onClick={this.pressButton}> Submit </button>
            </form>

            { this.state.isVisibleAll &&
            <div>
              <hr/>
              {/* by WGAN */}
              <h3>by WGAN</h3>
              <div className="imgPreview">
                {imagePreview}
              </div>

              {/* Text for model prediction */}
              <div>
                { this.state.imagePrediction &&
                  <p>The model predicted: {this.state.imagePrediction}
                  </p>
                }
              </div>

              { this.state.isVisible &&
                <img src={panda1} alt="panda training 1" />
              }
              <div>
                <Spinner size={60} spinnerColor={"#eb3489"} spinnerWidth={5} visible={this.state.isVisible} />
              </div>

              {/* Show result */}
              { !this.state.isHidden &&
              <div>
                <ShowWGANResult/>

                {/* Download CSV file */}
                <br/><br/>
                <CSVLink
                  filename={"wgan_2019-09-15_20000.csv"}
                  className="btn btn-primary"
                  data={sampleData_gan}
                  target="_blank"
                >
                  Download the generated data
                </CSVLink>
                <br/><br/><br/>
              </div>
              }

              <hr/>
              {/* by SMOTE */}
              <h3>by SMOTE</h3>

              { this.state.isVisible2 &&
                <img src={tiger1} alt="tiger training 1" />
              }
              <div>
                <Spinner size={60} spinnerColor={"#eb3489"} spinnerWidth={5} visible={this.state.isVisible2} />
              </div>

              {/* Show result */}
              { !this.state.isHidden2 &&
              <div>
                <ShowSMOTEResult/>

                {/* Download CSV file */}
                <br/><br/>
                <CSVLink
                  filename={"smote_2019-09-15_100_5_20000.csv"}
                  className="btn btn-primary"
                  data={sampleData_smote}
                  target="_blank"
                >
                  Download the generated data
                </CSVLink>
                <br/>
              </div>
              }
              </div>
            }
          </div>
        )
      }
    }

  return <FormBox/>

};


/**
 * These are 3rd pages - SMOTE
 */
const Smote = ({ location }) => {
    const onMatchedRoutes = (matchedRoutes) => {
      return [
        {
          route: {
            path: `${rootPath}/`,
            breadcrumbName: 'Home'
          }
        },
        ...matchedRoutes
      ];
    };

    class FormBox extends React.Component{

      // Constructor
      constructor() {
        super()
        this.state = {
          csvData: '',
          isVisible: false,
          isHidden: true,
          retData: [],
        }
        this.getInitialState = this.getInitialState.bind(this)
        this.pressButton2 = this.pressButton2.bind(this)
        this.getPhoto2 = this.getPhoto2.bind(this)
      }

      getInitialState() {
        return {
          file: '',
          imagePreviewUrl: ''
        }
      }

      pressButton2(e) {
        e.preventDefault();
        // TODO: do something with -> this.state.file

        var self = this;
        const formData = new FormData()
        formData.append('file', this.state.imageFile, 'img.png')
    
        var t0 = performance.now();
        axios({url:"http://127.0.0.1:5000/smoteclassifier", method:'get', timeout:240000})
        .then(function(response, data) {
            data = response.data;
            self.setState({retData:data})
            self.setState({isVisible:false})
            self.setState({isHidden: false})
            var t1 = performance.now();
            console.log("The time it took to inference by SMOTE-Classifier " + (t1 - t0) + " milliseconds.")
            console.log(response);
        })
        .catch(function (error) {
            console.log('Error:',error.message);
        })

        this.setState({
          isVisible: true
        })
        console.log('handle uploading-', this.state.file);
      }

      getPhoto2(e) {
        e.preventDefault();

        let reader = new FileReader();
        let file = e.target.files[0];

        this.setState({imageFile: file})

        reader.onloadend = () => {
          this.setState({
            file: file,
            imagePreviewUrl: reader.result
          });
        }

        reader.readAsDataURL(file);
      }

      render() {
        let {imagePreviewUrl} = this.state;
        let imagePreview = null;
        if (imagePreviewUrl) {
          imagePreview = (<img alt="..." src={imagePreviewUrl} />);
        } else {
          imagePreview = (<div className="previewText">Please select a dataset.</div>);
        }

        const ShowRetData = () =>
        <div>
          {
            Object.entries(this.state.retData)
            .map( ([key, value]) => <div><i>${key} is {value}</i><br/></div> )
          }
        </div>

        return (
          <div>
            <h1 className="py-3">SMOTE</h1>
            <Breadcrumb
              locationPath={location.pathname}
              onMatchedRoutes={onMatchedRoutes}
            />

            <h3>Input data to re-tran the classifier</h3>
            <form action='.' enctype="multipart/form-data">
              <input type='file'  onChange={this.getPhoto2}/>
              <button onClick={this.pressButton2}> Submit </button>
            </form>
            <div className="imgPreview">
              {imagePreview}
            </div>

            { this.state.isVisible &&
              <img src={tiger1} alt="tiger training 1" />
            }
            <div>
              <Spinner size={60} spinnerColor={"#eb3489"} spinnerWidth={5} visible={this.state.isVisible} />
            </div>

            { !this.state.isHidden &&
            <div>
            <br/><br/>
            <h3>Result of new classifier</h3>
            <ShowRetData/>
            </div>
            }

          </div>
        )
      }
    }

  return <FormBox/>

};


/**
 * These are 4th pages - WGAN
 */
const ZetaGAN = ({ location }) => {
    const onMatchedRoutes = (matchedRoutes) => {
      return [
        {
          route: {
            path: `${rootPath}/`,
            breadcrumbName: 'Home'
          }
        },
        ...matchedRoutes
      ];
    };

    class FormBox extends React.Component{

      // Constructor
      constructor() {
        super()
        this.state = {
          csvData: '',
          isVisible: false,
          isHidden: true,
          retData: [],
        }
        this.getInitialState = this.getInitialState.bind(this)
        this.pressButton = this.pressButton.bind(this)
        this.getPhoto = this.getPhoto.bind(this)
      }

      getInitialState() {
        return {
          file: '',
          imagePreviewUrl: ''
        }
      }

      pressButton(e) {
        e.preventDefault();
        // TODO: do something with -> this.state.file

        var self = this;
        const formData = new FormData()
        formData.append('file', this.state.imageFile, 'img.png')
    
        var t0 = performance.now();
        axios({url:"http://127.0.0.1:5000/ganclassifier", method:'get', timeout:240000})
        .then(function(response, data) {
            data = response.data;
            self.setState({retData:data})
            self.setState({isVisible:false})
            self.setState({isHidden:false})
            var t1 = performance.now();
            console.log("The time it took to inference by WGAN-Classifier " + (t1 - t0) + " milliseconds.")
            console.log(response);
        })
        .catch(function (error) {
            console.log('Error:',error.message);
        })

        this.setState({
          isVisible: true
        })
        console.log('handle uploading-', this.state.file);
      }

      getPhoto(e) {
        e.preventDefault();

        let reader = new FileReader();
        let file = e.target.files[0];

        this.setState({imageFile: file})
        this.setState({isVisible: false})
        this.setState({isHidden: true})

        reader.onloadend = () => {
          this.setState({
            file: file,
            imagePreviewUrl: reader.result
          });
        }

        reader.readAsDataURL(file);
      }

      render() {
        let {imagePreviewUrl} = this.state;
        let imagePreview = null;
        if (imagePreviewUrl) {
          imagePreview = (<img alt="..." src={imagePreviewUrl} />);
        } else {
          imagePreview = (<div className="previewText">Please select a dataset.</div>);
        }

        const ShowRetData = () =>
        <div>
          {
            Object.entries(this.state.retData)
            .map( ([key, value]) => <div><i>${key} is {value}</i><br/></div> )
          }
        </div>

        return (
          <div>
            <h1 className="py-3">WGAN</h1>
            <Breadcrumb
              locationPath={location.pathname}
              onMatchedRoutes={onMatchedRoutes}
            />

            <h3>Input data to re-tran the classifier</h3>
            <form action='.' enctype="multipart/form-data">
              <input type='file'  onChange={this.getPhoto}/>
              <button onClick={this.pressButton}> Submit </button>
            </form>
            <div className="imgPreview">
              {imagePreview}
            </div>

            { this.state.isVisible &&
              <img src={panda1} alt="panda training 1" />
            }
            <div>
              <Spinner size={60} spinnerColor={"#eb3489"} spinnerWidth={5} visible={this.state.isVisible} />
            </div>

            { !this.state.isHidden &&
              <div>
                <br/><br/>
                <h3>Result of new classifier</h3>
                <ShowRetData/>
              </div>
            }
          </div>
        )
      }
    }

  return <FormBox/>

};


/**
 * These are 5th pages
 */
const Summary = ({ location }) => {
    const onMatchedRoutes = (matchedRoutes) => {
      return [
        {
          route: {
            path: `${rootPath}/`,
            breadcrumbName: 'Home'
          }
        },
        ...matchedRoutes
      ];
    };

    class FormBox extends React.Component{

      // Constructor
      constructor() {
        super()
        this.state = {
          csvData: '',
        }
      }

      render() {
        return (
          <div>
            <h1 className="py-3">Summary</h1>
            <Breadcrumb
              locationPath={location.pathname}
              onMatchedRoutes={onMatchedRoutes}
            />

            <h3>SMOTE vs WGAN</h3>
              <CsvToHtmlTable
                data={perf}
                csvDelimiter=","
                tableClassName="table table-striped table-hover"
              />
              <img src={get_started} alt="get started" />
          </div>
        )
      }
    }

  return <FormBox/>
};


export { Home, Books, Smote, ZetaGAN, Summary };
