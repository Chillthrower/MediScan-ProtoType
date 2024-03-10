import React, { useRef } from 'react';
import { BrowserRouter as Router, Route } from 'react-router-dom';
import PatientSignupForm from './components/PatientSignupForm';
import DoctorSignupForm from './components/DoctorSignupForm';
import DoctorLoginForm from './components/DoctorLoginForm';
import PatientLoginForm from './components/PatientLoginForm';

const App = () => {
    const homeRef = useRef(null);
    const SignupasRef = useRef(null)
    const capture_or_uploadRef = useRef(null)

    return (
        <Router>
            <Route exact path="/patient_signup" component={PatientSignupForm} />
            <Route exact path="/patient_login" component={PatientLoginForm} />
            <Route exact path="/doctor_signup" component={DoctorSignupForm} />
            <Route exact path="/doctor_login" component={DoctorLoginForm} />
            <Route
                path="/"
                render={() => (
                    <iframe
                        ref={SignupasRef}
                        title="Signupas"
                        style={{ width: '100%', height: '100vh', border: 'none' }}
                        onLoad={() => {
                            SignupasRef.current.contentWindow.location.reload();
                        }}
                        src="/templates/signupas.html"
                    />
                )}
            />
            <Route
                path="/"
                render={() => (
                    <iframe
                        ref={capture_or_uploadRef}
                        title="capture_or_upload"
                        style={{ width: '100%', height: '100vh', border: 'none' }}
                        onLoad={() => {
                            capture_or_uploadRef.current.contentWindow.location.reload();
                        }}
                        src="/templates/capture_or_uploadRef.html"
                    />
                )}
            />
            <Route
                path="/"
                render={() => (
                    <iframe
                        ref={homeRef}
                        title="Home"
                        style={{ width: '100%', height: '100vh', border: 'none' }}
                        onLoad={() => {
                            homeRef.current.contentWindow.location.reload();
                        }}
                        src="/templates/home.html"
                    />
                )}
            />
        </Router>
    );
};

export default App;
