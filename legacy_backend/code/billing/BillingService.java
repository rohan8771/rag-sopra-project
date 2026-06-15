package billing;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;


public class BillingService {

    private final String dbUrl;
    private final String dbUser;
    private final String dbPassword;

    /*
     * Constructor receives database connection details.
     * In a real legacy app, these may come from application.properties.
     */
    public BillingService(String dbUrl, String dbUser, String dbPassword) {
        this.dbUrl = dbUrl;
        this.dbUser = dbUser;
        this.dbPassword = dbPassword;
    }

    /*
     * Updates payment status for one invoice.
     */
    public void updatePaymentStatus(int invoiceId, String status) throws SQLException {

        /*
         * Legacy issue:
         * SQL is manually written inside Java code.
         */
        String sql = "UPDATE INVOICE SET STATUS = '" + status + "' WHERE ID = " + invoiceId;

        /*
         * Legacy issue:
         * Statement + string concatenation can cause SQL injection.
         *
         * Modern approach:
         * Use PreparedStatement, repository layer, ORM, validation, etc.
         */
        try (
            Connection connection = DriverManager.getConnection(dbUrl, dbUser, dbPassword);
            Statement statement = connection.createStatement()
        ) {
            int rowsUpdated = statement.executeUpdate(sql);

            System.out.println("Rows updated: " + rowsUpdated);
        }
    }
}