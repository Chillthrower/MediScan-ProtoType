function ValidateAge() {
    var age=document.getElementById("age").value;
    var message=document.getElementById("error_message");
    
    
    if(age>0 && age<100 ){
        message.innerHTML="Valid";
        message.style.color="green";
    }
    else{
        message.innerHTML="Invalid";
        message.style.color="red";
    }
}