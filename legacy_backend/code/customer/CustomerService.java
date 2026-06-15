package customer;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public class CustomerService {

    private final String dbUrl;
    private final String dbUser;
    private final String dbPassword;

    /*
     * Stores database connection details.
     */
    public CustomerService(String dbUrl, String dbUser, String dbPassword) {
        this.dbUrl = dbUrl;
        this.dbUser = dbUser;
        this.dbPassword = dbPassword;
    }

    /*
     * Looks up a customer's name by customer ID.
     */
    public String getCustomerName(int customerId) throws SQLException {

        /*
         * Legacy issue:
         * SQL is written directly inside service code.
         */
        String sql = "SELECT NAME FROM CUSTOMER WHERE ID = " + customerId;

        /*
         * Opens database connection and executes query.
         *
         * Legacy issue:
         * Uses raw JDBC Statement instead of safer PreparedStatement.
         */
        try (
            Connection connection = DriverManager.getConnection(dbUrl, dbUser, dbPassword);
            Statement statement = connection.createStatement();
            ResultSet resultSet = statement.executeQuery(sql)
        ) {
            if (resultSet.next()) {
                return resultSet.getString("NAME");
            }

            return null;
        }
    }
}