CREATE TABLE expense (
    expenseid SERIAL PRIMARY KEY,
    userid INT NOT NULL REFERENCES users(userid),
    mode_of_payment VARCHAR(50) NOT NULL,
    amount NUMERIC(10, 2) NOT NULL,
    credit_or_debit VARCHAR(50) NOT NULL,
    expense_date DATE NOT NULL,
    description VARCHAR(255) NOT NULL
);