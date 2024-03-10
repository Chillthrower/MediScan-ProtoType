import React, { useRef } from 'react';
import axios from 'axios';

const DoctorSignupForm = () => {
    const nameRef = useRef(null);
    const ageRef = useRef(null);
    const genderRef = useRef(null);
    const emailRef = useRef(null);
    const countryRef = useRef(null);
    const phoneNumberRef = useRef(null);
    const addressRef = useRef(null);
    const pincodeRef = useRef(null);
    const passwordRef = useRef(null);
    const confirmPasswordRef = useRef(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        const formData = {
            name: nameRef.current.value,
            age: ageRef.current.value,
            gender: genderRef.current.value,
            email: emailRef.current.value,
            country: countryRef.current.value,
            phone_number: phoneNumberRef.current.value,
            address: addressRef.current.value,
            pincode: pincodeRef.current.value,
            password: passwordRef.current.value,
            confirmPassword: confirmPasswordRef.current.value
        };
        try {
            const response = await axios.post('doctor_signup/', formData);
            console.log(response.data);
            // Redirect to home page or handle success
        } catch (error) {
            console.error('Signup failed:', error);
            // Handle failure, maybe show error messages to the user
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <h1 className="heading">Signup section for doctors</h1>
            <p>Name:<input type="text" ref={nameRef} className="name" placeholder="Enter your name" /></p>
            <p>Age:<input type="number" ref={ageRef} className="age" id="age" /></p>
            <p>Gender:</p>
            <p>
                Male<input type="radio" ref={genderRef} name="gender" value="M" id="male" />
                Female<input type="radio" ref={genderRef} name="gender" value="F" id="female" />
                Other<input type="radio" ref={genderRef} name="gender" value="O" id="other" />
            </p>
            <p>Email:<input type="email" ref={emailRef} className="email" placeholder="Enter your email" /></p>
            <p>Country:<input type="text" ref={countryRef} className="country" placeholder="Enter your country" /></p>
            <p>Phone Number:<input type="tel" ref={phoneNumberRef} className="phone" placeholder="Enter your phone number" /></p>
            <p>Address:<textarea ref={addressRef} className="address" placeholder="Enter your address"></textarea></p>
            <p>Pincode:<input type="number" ref={pincodeRef} className="pincode" placeholder="Enter your pincode" /></p>
            <p>Password:<input type="password" ref={passwordRef} className="password" placeholder="Enter your password" /></p>
            <p>Confirm Password:<input type="password" ref={confirmPasswordRef} className="confirm-password" placeholder="Confirm your password" /></p>
            <input type="submit" className="submit" value="Submit" />
        </form>
    );
};

export default DoctorSignupForm;
