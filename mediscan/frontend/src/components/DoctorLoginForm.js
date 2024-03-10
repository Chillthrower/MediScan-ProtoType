import React, { useRef } from 'react';
import axios from 'axios';

const DoctorLoginForm = () => {
    const emailRef = useRef(null);
    const passwordRef = useRef(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        const formData = {
            email: emailRef.current.value,
            password: passwordRef.current.value
        };
        try {
            const response = await axios.post('doctor_login/', formData);
            console.log(response.data);
            // Handle login success (redirect or show success message)
        } catch (error) {
            console.error('Login failed:', error);
            // Handle login failure (show error message)
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <h1 className="heading">Doctor Login</h1>
            <p>Email:<input type="email" ref={emailRef} className="email" placeholder="Enter your email" /></p>
            <p>Password:<input type="password" ref={passwordRef} className="password" placeholder="Enter your password" /></p>
            <input type="submit" className="submit" value="Login" />
        </form>
    );
};

export default DoctorLoginForm;
