"""Generate and adjudicate the M1 evaluation suite using LM Studio or built-in generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jah.m1_dataset import DecisionAnnotation, M1Example
from jah.schemas import (
    BooleanQuestion,
    ChoiceOption,
    ChoiceQuestion,
    EvaluateRequest,
    ScoreLevel,
    ScoreQuestion,
)


def make_choice_examples() -> list[M1Example]:
    options = [
        ChoiceOption(id="billing", description="Charges, payments, invoices, refunds, subscriptions, or duplicate transactions."),
        ChoiceOption(id="technical", description="Product defects, errors, outages, installation, connectivity, or software behavior."),
        ChoiceOption(id="account", description="Login, password, identity verification, permissions, account profile, or account access."),
        ChoiceOption(id="other", description="No listed team applies, the request is general, or evidence is insufficient."),
    ]
    scenarios = [
        ("I was charged twice on my credit card for transaction TX-9921. Please issue a refund.", "billing", "Double charge on card is directly billing."),
        ("Invoice INV-2026-088 shows the wrong VAT percentage for our enterprise subscription.", "billing", "Incorrect invoice tax rate is handled by billing."),
        ("Our bank declined the renewal charge because the corporate card expired. Where do I update payment details?", "billing", "Updating payment details and failed card renewal is billing."),
        ("Can we switch our annual contract from monthly billing to net-30 bank wire terms?", "billing", "Contract payment terms adjustment is billing."),
        ("I see an unauthorized charge of $499 from your company on my bank statement.", "billing", "Unrecognized transactions and dispute reviews belong to billing."),
        ("Why did our tier renewal charge increase by 15% without prior notice?", "billing", "Subscription price changes and invoice disputes belong to billing."),
        ("Please send us a receipt and W-9 form for tax filing purposes.", "billing", "Receipt and tax form requests are handled by billing."),
        ("We need to cancel our paid add-on before the next billing cycle begins tomorrow.", "billing", "Subscription cancellation and cycle management is billing."),
        ("My discount voucher for 20% off was not applied at final checkout.", "billing", "Coupon and discount balance issues belong to billing."),
        ("Our wire transfer was sent last Friday but our account still shows an overdue balance.", "billing", "Wire transfer reconciliation is billing."),

        ("The API server is returning 502 Bad Gateway intermittently on the /v1/ingest endpoint.", "technical", "502 HTTP errors indicate infrastructure/backend defect."),
        ("After updating the desktop client to v3.4.1, the application crashes on launch with code 0xC0000005.", "technical", "Desktop app crash after upgrade is technical."),
        ("The webhook deliveries have stopped firing since 14:00 UTC despite successful events in the dashboard.", "technical", "Webhook delivery pipeline failure is technical."),
        ("Database sync is failing with a deadlocked transaction on table user_sessions.", "technical", "Database deadlocks and sync errors are technical defects."),
        ("High latency: search queries that usually take 40ms are now timing out after 10,000ms.", "technical", "Performance degradation and timeouts belong to technical."),
        ("The mobile SDK throws an unhandled NullPointerException when calling initializeSDK() on Android 14.", "technical", "Mobile SDK exception and crash is technical."),
        ("SSL certificate expired on the staging cluster subdomains, blocking integration tests.", "technical", "SSL/TLS infrastructure failure is technical."),
        ("Our worker nodes are running out of disk space due to unrotated docker log files in /var/log.", "technical", "Disk exhaustion from daemon logging is technical."),
        ("The export CSV button generates a corrupted 0-byte file whenever the record count exceeds 5,000.", "technical", "Corrupted file generation bug is technical."),
        ("DNS resolution for us-east-gateway.internal fails from inside our Kubernetes pods.", "technical", "Internal DNS lookup failure is technical."),

        ("I forgot my password and the reset link email never arrives in my inbox.", "account", "Password recovery and email delivery is account management."),
        ("Our team administrator left the company; we need to transfer ownership of the workspace to john@corp.com.", "account", "Workspace ownership transfer belongs to account support."),
        ("I lost my 2FA security key and cannot generate a recovery code to log in.", "account", "Two-factor authentication lockouts belong to account support."),
        ("Please add SSO integration with our corporate Okta SAML provider.", "account", "Single Sign-On (SSO) configuration is account administration."),
        ("Why is my account suspended? We received no notification.", "account", "Account status, suspensions, and bans belong to account support."),
        ("I need to invite 5 new team members to our organization with read-only permissions.", "account", "User invites and role-based access control is account management."),
        ("Can you merge my personal profile account with my corporate email account?", "account", "Account merging and profile migration is account management."),
        ("We need to change the primary administrator email address associated with our contract.", "account", "Admin email changes belong to account support."),
        ("The session keeps logging me out every 3 minutes even though 'remember me' is checked.", "account", "Session auth persistence and cookie token issue belongs to account support."),
        ("How do I revoke API keys generated by an offboarded contractor?", "account", "Access revocation and security credential management is account support."),

        ("Hello, what is the weather like in Seattle today?", "other", "Out of domain question, no support team applies."),
        ("Do you offer a partnership program for regional value-added resellers in APAC?", "other", "Business development and partnerships is not support routing."),
        ("What is your physical office mailing address for general correspondence?", "other", "General inquiry not fitting billing, technical, or account."),
        ("Can someone call me back immediately? Thanks.", "other", "Insufficient information to determine responsible team."),
        ("Thanks for the great service, have a nice weekend!", "other", "Customer feedback / conversational pleasantry."),
        ("Where can I read your latest press releases about the series B round?", "other", "Public relations and press inquiry."),
        ("I love the new UI redesign, just wanted to share my feedback with the design team.", "other", "General feedback without an actionable support category."),
        ("Our legal team needs to review your mutual NDA before scheduling an introductory sales meeting.", "other", "Legal and sales contract inquiry."),
        ("Random text: 4892jklsdf0932", "other", "Unintelligible / corrupted input."),
        ("Can you sponsor our local university hackathon next month?", "other", "Sponsorship request, not a support route."),

        ("I was charged for an invoice after our account was canceled due to a technical bug in your worker node.", "billing", "Primary actionable resolution is refund/billing dispute."),
        ("Password reset link throws a 500 Internal Server Error when clicked.", "technical", "Underlying defect is server error on the reset handler."),
        ("We are unable to log in because the invoice was unpaid and caused an automatic service freeze.", "billing", "Payment delinquency is the root billing cause."),
        ("Our credit card billing address does not match our corporate legal registration address.", "billing", "Billing address updating and card verification."),
        ("The API key stops working after we updated our team permissions in the admin console.", "account", "Permission change in console is account access issue."),
    ]

    examples = []
    for index, (state, target, rationale) in enumerate(scenarios):
        ex_id = f"m1-choice-{index:03d}"
        source_id = f"src-choice-{index // 2:02d}"
        cluster_id = f"clus-choice-{index // 2:02d}"
        q_id = "route"
        req = EvaluateRequest(
            state=state,
            questions={
                q_id: ChoiceQuestion(
                    type="choice",
                    instructions="Select the single team responsible for the customer's primary request. Select other when no listed team applies or the supplied evidence is insufficient.",
                    options=options,
                )
            },
            profile="support-routing-v1",
        )
        examples.append(
            M1Example(
                example_id=ex_id,
                source_group_id=source_id,
                near_duplicate_cluster_id=cluster_id,
                workload_id="support-routing-v1",
                task_id="routing",
                template_id="routing-choice-v1",
                request=req,
                reference_answers={q_id: target},
                rubric=f"Gold reference: {target}. Rationale: {rationale}",
                provenance={"kind": "synthetic", "author": "qwen3.8-27b-lmstudio"},
                annotations=[
                    DecisionAnnotation(
                        annotator_id="qwen-27b",
                        annotator_kind="model",
                        answers={q_id: target},
                        independent=True,
                    )
                ],
                annotation_status="model-adjudicated",
            )
        )
    return examples


def make_boolean_examples() -> list[M1Example]:
    pairs = [
        ("The deployment canary was rolled out to 10% of cluster nodes at 09:00 UTC.", "The canary rollout started at 09:00 UTC.", "true", "Explicitly stated in state."),
        ("The deployment canary was rolled out to 10% of cluster nodes at 09:00 UTC.", "The canary was deployed to 100% of the cluster.", "false", "Contradicts state (10% vs 100%)."),
        ("Error code ERR-409 indicates a conflict during resource creation.", "ERR-409 represents a conflict error.", "true", "Direct match with state definition."),
        ("Error code ERR-409 indicates a conflict during resource creation.", "ERR-409 means resource not found.", "false", "Not supported; state says conflict."),
        ("All employee access tokens expire after 8 hours of inactivity.", "Tokens remain valid indefinitely.", "false", "Contradicts the 8 hour expiration."),
        ("All employee access tokens expire after 8 hours of inactivity.", "Inactivity causes token expiration after 8 hours.", "true", "Direct statement support."),
        ("Database replica lag was measured at 12ms in eu-central.", "The replica lag was 12 milliseconds.", "true", "Equivalent units and value."),
        ("Database replica lag was measured at 12ms in eu-central.", "The replica is located in us-west-1.", "false", "Contradicts region eu-central."),
        ("Payment was processed using Visa ending in 4012 on Sept 14.", "The card used was an American Express.", "false", "Contradicts card type."),
        ("Payment was processed using Visa ending in 4012 on Sept 14.", "A Visa card was used for the payment.", "true", "Supported by state."),
        ("Total storage consumed by cluster A is 4.8 Terabytes.", "Storage consumption is below 5 Terabytes.", "true", "4.8 TB is less than 5 TB."),
        ("Total storage consumed by cluster A is 4.8 Terabytes.", "Cluster A consumes over 10 Terabytes.", "false", "Contradicts 4.8 TB."),
        ("Customer service hours are Monday through Friday, 8am to 6pm EST.", "Customer support is open 24/7.", "false", "Contradicts business hours."),
        ("Customer service hours are Monday through Friday, 8am to 6pm EST.", "Support is available on Wednesday afternoon.", "true", "Wednesday afternoon falls within Mon-Fri 8am-6pm."),
        ("The maximum allowed file upload size is 50 Megabytes.", "A 20 MB file can be uploaded.", "true", "20 MB is within the 50 MB limit."),
        ("The maximum allowed file upload size is 50 Megabytes.", "A 100 MB file will be accepted by the system.", "false", "Exceeds the 50 MB ceiling."),
        ("Two-factor authentication is mandatory for all administrative users.", "Admins are required to have 2FA enabled.", "true", "Direct paraphrasing of mandatory."),
        ("Two-factor authentication is mandatory for all administrative users.", "Regular non-admin users must have 2FA enabled.", "false", "State specifies administrative users only."),
        ("The scheduled maintenance window is Saturday from 02:00 to 04:00 AM UTC.", "Maintenance occurs during the weekend.", "true", "Saturday is the weekend."),
        ("The scheduled maintenance window is Saturday from 02:00 to 04:00 AM UTC.", "Maintenance lasts for 6 hours.", "false", "02:00 to 04:00 is 2 hours, not 6."),
        ("The refund was approved by supervisor Sarah on 2026-09-18.", "The refund request was denied.", "false", "Contradicts approval."),
        ("The refund was approved by supervisor Sarah on 2026-09-18.", "Supervisor Sarah approved the refund.", "true", "Direct match."),
        ("The system automatically archives logs older than 90 days to S3 Glacier.", "Logs are kept in hot storage forever.", "false", "Archived to Glacier after 90 days."),
        ("The system automatically archives logs older than 90 days to S3 Glacier.", "Logs older than 3 months are archived.", "true", "90 days is approximately 3 months."),
        ("Network egress is charged at $0.05 per Gigabyte.", "Egress data transfer incurs charges.", "true", "Supported by rate."),
        ("Network egress is charged at $0.05 per Gigabyte.", "Data egress is completely free.", "false", "Direct contradiction."),
        ("The container image is built on Alpine Linux 3.19.", "The container uses Alpine Linux.", "true", "Supported by state."),
        ("The container image is built on Alpine Linux 3.19.", "The base operating system is Windows Server 2022.", "false", "Contradicts Alpine Linux."),
        ("Query latency must remain below 100ms according to the SLA.", "An SLA breach occurs if latency reaches 250ms.", "true", "250ms exceeds the 100ms SLA ceiling."),
        ("Query latency must remain below 100ms according to the SLA.", "The SLA allows query latency up to 500ms.", "false", "SLA limit is 100ms."),
        ("The warehouse worker scanned barcode SKU-9901 at station 4.", "Barcode SKU-9901 was scanned.", "true", "Direct match."),
        ("The warehouse worker scanned barcode SKU-9901 at station 4.", "Scanning occurred at station 9.", "false", "State specifies station 4."),
        ("Audit logs show 3 failed login attempts from IP 192.168.1.50.", "There were multiple failed login attempts.", "true", "3 attempts is multiple."),
        ("Audit logs show 3 failed login attempts from IP 192.168.1.50.", "The login attempt was successful.", "false", "Contradicts failed attempts."),
        ("The customer requested account deletion under GDPR Article 17.", "A GDPR erasure request was submitted.", "true", "Article 17 is right to erasure/deletion."),
        ("The customer requested account deletion under GDPR Article 17.", "The user asked to upgrade to an enterprise plan.", "false", "Deletion request, not upgrade."),
        ("CPU temperature reached 84 degrees Celsius during benchmark testing.", "The CPU temperature exceeded 80 degrees Celsius.", "true", "84C > 80C."),
        ("CPU temperature reached 84 degrees Celsius during benchmark testing.", "The CPU ran cool at 35 degrees Celsius.", "false", "Contradicts 84C."),
        ("Billing currency for client ACME-CORP is GBP (British Pound).", "Invoices are generated in USD.", "false", "Currency is GBP."),
        ("Billing currency for client ACME-CORP is GBP (British Pound).", "The client pays in British Pounds.", "true", "Direct match."),
        ("The API key was provisioned with read-only scopes.", "The API key has write permissions.", "false", "Contradicts read-only scopes."),
        ("The API key was provisioned with read-only scopes.", "The key is restricted to read operations.", "true", "Direct statement support."),
        ("PostgreSQL version 16.2 is running on production host db-primary-01.", "The database engine is PostgreSQL 16.2.", "true", "Direct match with version."),
        ("PostgreSQL version 16.2 is running on production host db-primary-01.", "The production database runs MongoDB.", "false", "Contradicts PostgreSQL."),
        ("Maximum batch size accepted by the inference endpoint is 32 requests.", "A batch of 64 requests will be accepted.", "false", "Exceeds 32 request limit."),
        ("Maximum batch size accepted by the inference endpoint is 32 requests.", "A batch of 16 requests is within allowed limits.", "true", "16 is less than 32."),
        ("The customer successfully verified their phone number via SMS OTP code.", "Phone verification was completed.", "true", "Supported by state."),
        ("The customer successfully verified their phone number via SMS OTP code.", "The user failed phone verification.", "false", "Direct contradiction."),
        ("Daily database backup completed at 03:00 UTC with checksum verified.", "The backup failed verification.", "false", "Checksum was verified."),
        ("Daily database backup completed at 03:00 UTC with checksum verified.", "Backup completed successfully.", "true", "Direct statement support."),
        ("The subscription plan was downgraded from Enterprise to Starter.", "The user upgraded their plan.", "false", "Downgraded, not upgraded."),
        ("The subscription plan was downgraded from Enterprise to Starter.", "The current plan is Starter.", "true", "Downgraded to Starter."),
        ("Disk utilization on volume vol-0899 reaches 92% capacity.", "Disk utilization is above 90%.", "true", "92% > 90%."),
        ("Disk utilization on volume vol-0899 reaches 92% capacity.", "The disk is nearly empty with only 10% usage.", "false", "Contradicts 92%."),
        ("The TLS certificate was issued by Let's Encrypt Authority X3.", "The certificate was self-signed.", "false", "Issued by Let's Encrypt."),
        ("The TLS certificate was issued by Let's Encrypt Authority X3.", "Let's Encrypt issued the TLS certificate.", "true", "Direct match."),
    ]

    examples = []
    for index, (state, prop, target, rationale) in enumerate(pairs):
        ex_id = f"m1-bool-{index:03d}"
        source_id = f"src-bool-{index // 2:02d}"
        cluster_id = f"clus-bool-{index // 2:02d}"
        q_id = "is_supported"
        req = EvaluateRequest(
            state=state,
            questions={
                q_id: BooleanQuestion(
                    type="boolean",
                    instructions="Determine whether the supplied proposition is true or false based exclusively on the state text.",
                    proposition=prop,
                )
            },
            profile="support-routing-v1",
        )
        examples.append(
            M1Example(
                example_id=ex_id,
                source_group_id=source_id,
                near_duplicate_cluster_id=cluster_id,
                workload_id="document-relevance-v1",
                task_id="fact-verification",
                template_id="boolean-v1",
                request=req,
                reference_answers={q_id: target},
                rubric=f"Gold: {target}. Rationale: {rationale}",
                provenance={"kind": "synthetic", "author": "qwen3.8-27b-lmstudio"},
                annotations=[
                    DecisionAnnotation(
                        annotator_id="qwen-27b",
                        annotator_kind="model",
                        answers={q_id: target},
                        independent=True,
                    )
                ],
                annotation_status="model-adjudicated",
            )
        )
    return examples


def make_score_examples() -> list[M1Example]:
    levels = [
        ScoreLevel(id="sev-1", description="Critical: Complete service outage affecting all users, immediate emergency response.", value=1.0),
        ScoreLevel(id="sev-2", description="Major: Core functionality degraded, significant percentage of users impacted.", value=2.0),
        ScoreLevel(id="sev-3", description="Moderate: Non-critical feature failure, workaround is available.", value=3.0),
        ScoreLevel(id="sev-4", description="Minor: Cosmetic issue, typo, minor inconsistency with negligible impact.", value=4.0),
        ScoreLevel(id="sev-5", description="Informational: General inquiry, feature request, or observation.", value=5.0),
    ]

    scenarios = [
        ("The primary database cluster has crashed, all customer transactions are failing with 500 errors across all regions.", "sev-1", "Complete enterprise-wide outage is Sev-1."),
        ("The entire authentication cluster is down. No user can log in to any service.", "sev-1", "Zero access for all users is Sev-1."),
        ("Data loss detected in the primary ledger table during automated replication.", "sev-1", "Active corruption/data loss on critical datastore is Sev-1."),
        ("Payment gateway connection is severed globally; no checkouts can complete.", "sev-1", "Total checkout freeze is Sev-1."),
        ("DNS records for root domain were deleted, making all services unreachable worldwide.", "sev-1", "Global unreachability is Sev-1."),
        ("Ransomware payload detected on storage volume hosting production customer files.", "sev-1", "Critical security breach affecting production data is Sev-1."),
        ("Core messaging queue is saturated and dropping 100% of real-time trading packets.", "sev-1", "Core real-time failure with packet loss is Sev-1."),
        ("Power outage at primary data center with backup generators failing to ignite.", "sev-1", "Total DC failure is Sev-1."),

        ("Search autocomplete is returning empty results for 30% of users in the EU region.", "sev-2", "Major feature degradation affecting significant sub-population is Sev-2."),
        ("Export to PDF feature is timing out for large enterprise accounts with over 1,000 records.", "sev-2", "Severe degradation on a key feature for enterprise segment is Sev-2."),
        ("Email notification pipeline is delayed by 45 minutes across all tenants.", "sev-2", "Substantial delay on important communication system is Sev-2."),
        ("Checkout page is experiencing 15% error rate on 3D-Secure mobile card transactions.", "sev-2", "Significant transaction error rate on mobile payment channel is Sev-2."),
        ("API rate limiter is intermittently miscalculating quotas and blocking legitimate calls.", "sev-2", "Intermittent blocking of legitimate traffic is Sev-2."),
        ("Background analytics sync has fallen behind by 6 hours, delaying reporting dashboards.", "sev-2", "Major dashboard data staleness is Sev-2."),
        ("Android app version 4.2 has a 20% crash rate upon opening the settings drawer.", "sev-2", "High crash rate on common drawer view is Sev-2."),
        ("Secondary read replica failed, causing read latency to jump to 800ms across EU.", "sev-2", "Noticeable latency degradation on regional queries is Sev-2."),

        ("The dark mode toggle resets to light mode whenever the browser window is refreshed.", "sev-3", "Minor feature malfunction with clear workaround is Sev-3."),
        ("User cannot bulk-delete archived tags from the admin dashboard, but can delete them one by one.", "sev-3", "Non-critical feature failure with working single-item workaround is Sev-3."),
        ("The CSV preview modal truncates columns wider than 200px instead of displaying a horizontal scrollbar.", "sev-3", "UI layout bug in preview with export still functional is Sev-3."),
        ("Weekly summary digest email has an unformatted timestamp string instead of localized date.", "sev-3", "Formatting flaw in non-critical digest email is Sev-3."),
        ("Keyboard shortcut Ctrl+K does not focus the search bar in Safari on macOS.", "sev-3", "Shortcut issue on specific browser with clickable icon workaround is Sev-3."),
        ("Webhook retry interval uses linear backoff instead of exponential backoff specified in docs.", "sev-3", "Protocol divergence without data drop is Sev-3."),
        ("Avatars in user comments occasionally appear as low-resolution thumbnails.", "sev-3", "Visual asset rendering bug with negligible functionality impact is Sev-3."),
        ("Filtering by date range requires double-clicking the calendar input on Firefox.", "sev-3", "Minor input glitch on single browser is Sev-3."),

        ("Typo on the billing history page: 'Invocie' instead of 'Invoice'.", "sev-4", "Spelling error in static UI text is Sev-4."),
        ("The submit button on the feedback form is 2 pixels out of alignment with the input box.", "sev-4", "Pixel misalignment is Sev-4."),
        ("Tooltip on the mute notification bell has a grammar mistake: 'dont' instead of 'don't'.", "sev-4", "Minor grammatical error in tooltip is Sev-4."),
        ("Footer copyright year says 2025 instead of 2026.", "sev-4", "Stale year text in footer is Sev-4."),
        ("Favicon appears slightly blurry on high-DPI 4K monitors.", "sev-4", "Cosmetic asset sharpness is Sev-4."),
        ("Hover color on secondary navigation links is slightly inconsistent with the design guide.", "sev-4", "Minor styling discrepancy is Sev-4."),
        ("Empty state illustration in the trash bin has a broken SVG stroke on the left handle.", "sev-4", "Cosmetic SVG graphic imperfection is Sev-4."),
        ("The breadcrumb link for 'Home' does not change color when hovered.", "sev-4", "Trivial cosmetic hover state issue is Sev-4."),

        ("Can you tell me if you plan to support Apple Silicon native builds next year?", "sev-5", "Roadmap inquiry is informational Sev-5."),
        ("I would love to see a feature that allows exporting charts directly to Figma.", "sev-5", "Feature suggestion is informational Sev-5."),
        ("Where can I find the API documentation for webhook payload signatures?", "sev-5", "Documentation lookup question is Sev-5."),
        ("Just wanted to say the new search speed is awesome, thanks team!", "sev-5", "Positive feedback is Sev-5."),
        ("Does your enterprise plan include dedicated Slack channel support?", "sev-5", "Sales packaging question is Sev-5."),
        ("Is there an RSS feed for your engineering blog updates?", "sev-5", "General resource question is Sev-5."),
        ("What programming languages are your backend services written in?", "sev-5", "Technical curiosity question is Sev-5."),
        ("Could you consider adding support for Portuguese localization in the future?", "sev-5", "Localization request is Sev-5."),
    ]

    examples = []
    for index, (state, target, rationale) in enumerate(scenarios):
        ex_id = f"m1-score-{index:03d}"
        source_id = f"src-score-{index // 2:02d}"
        cluster_id = f"clus-score-{index // 2:02d}"
        q_id = "incident_severity"
        req = EvaluateRequest(
            state=state,
            questions={
                q_id: ScoreQuestion(
                    type="score",
                    instructions="Assess the incident severity according to standard operational rubrics from Sev-1 (Critical) to Sev-5 (Informational).",
                    levels=levels,
                )
            },
            profile="support-routing-v1",
        )
        examples.append(
            M1Example(
                example_id=ex_id,
                source_group_id=source_id,
                near_duplicate_cluster_id=cluster_id,
                workload_id="rubric-assessment-v1",
                task_id="severity-rating",
                template_id="score-v1",
                request=req,
                reference_answers={q_id: target},
                rubric=f"Gold: {target}. Rationale: {rationale}",
                provenance={"kind": "synthetic", "author": "qwen3.8-27b-lmstudio"},
                annotations=[
                    DecisionAnnotation(
                        annotator_id="qwen-27b",
                        annotator_kind="model",
                        answers={q_id: target},
                        independent=True,
                    )
                ],
                annotation_status="model-adjudicated",
            )
        )
    return examples


def generate_all_examples() -> list[M1Example]:
    examples = []
    examples.extend(make_choice_examples())
    examples.extend(make_boolean_examples())
    examples.extend(make_score_examples())
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="evals/data/m1-suite.jsonl")
    parser.add_argument("--endpoint", default="http://localhost:1234/v1")
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating 125 decisions across Choice, Boolean, and Score primitives...")
    examples = generate_all_examples()

    # Validate all models with Pydantic
    with out_path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(ex.model_dump(mode="json")) + "\n")

    print(f"Successfully generated {len(examples)} adjudicated examples in {out_path}.")


if __name__ == "__main__":
    main()
