CREATE TABLE users (
    userid SERIAL PRIMARY KEY,
    employeeid INT NOT NULL,
    name VARCHAR(100) NOT NULL,
    dob DATE NOT NULL,
    contact_number VARCHAR(20) NOT NULL,
    email VARCHAR(100)
);