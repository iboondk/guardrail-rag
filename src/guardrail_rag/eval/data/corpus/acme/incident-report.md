# Incident Report INC-2041

On March 3, the Acme ingestion service stopped accepting uploads for 47 minutes. The root cause was an expired certificate on the internal load balancer. Automated renewal had been disabled during a migration and was not re-enabled afterwards.

Detection came from a synthetic check that failed three times in a row. The on-call engineer paged the platform team at 02:14, and the service recovered at 03:01 after the certificate was replaced.

Impact: about 1,200 uploads were rejected and customers received an error code. The team replayed the rejected uploads from the client retry queue, and 98 percent of them completed successfully. Two enterprise customers asked for credits, which were granted under the service level agreement.

Follow-up actions: certificate renewal is now monitored with alerts 30 days before expiry, migrations must list every certificate they touch, and the synthetic check runs from two regions.

The post-incident review was published to all customers within five business days. It lists the timeline, the decisions made during the outage, and the owners of each follow-up action.
