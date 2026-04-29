# Sanity-check sample of feasibility judge

Random sample (seed=42) of ~4 tasks per category from the v2 log. Visit each website and verify the judge's call. Note the judge's reasoning is the *only* thing that determines the classification — the HTTP status was used as input but not as the decision.

## WEBSITE_DOWN — HTTP 2xx (judge override of probe)  *(pool size: 181)*

- **arttesia.co.uk**  `HTTP 200 reachable=True`  `conf=1.00`  `→ Skipped`
  - *Task:* Determine if arttesia.co.uk has any functional content beyond its footer, specifically by ...
  - *Judge:* The browse tool was unable to access the content of arttesia.co.uk, preventing any further navigation or assessment of its functional content.

- **free-barcode.com**  `HTTP 200 reachable=True`  `conf=1.00`  `→ Skipped`
  - *Task:* Generate a Code128B barcode for the data 'PRODUCT-XYZ-123' with a width of 4 inches and a ...
  - *Judge:* The website free-barcode.com could not be accessed by the browsing tool, indicating it is currently unreachable.

- **cdhb.health.nz**  `HTTP 200 reachable=True`  `conf=1.00`  `→ Skipped`
  - *Task:* What are the visiting hours for Christchurch Hospital on the HealthInfo Canterbury website...
  - *Judge:* The browsing tool was unable to access the starting URL, indicating the website is unreachable from the tool's perspective.

- **porcinehealthmanagement.biomedcentral.com**  `HTTP 200 reachable=True`  `conf=1.00`  `→ Skipped`
  - *Task:* Find the median time from manuscript submission to acceptance for Porcine Health Managemen...
  - *Judge:* The browse tool was unable to access the provided URL, making it impossible to complete the task.

## WEBSITE_DOWN — HTTP 4xx/5xx  *(pool size: 235)*

- **ville-agde.fr**  `HTTP 403 reachable=True`  `conf=0.90`  `→ Skipped`
  - *Task:* Find the main phone number for the Office de Tourisme Cap d'Agde Méditerranée....
  - *Judge:* HTTP probe returned 403 Forbidden — site is blocking automated access.

- **hawaiianairlines.co.jp**  `HTTP 403 reachable=True`  `conf=0.90`  `→ Skipped`
  - *Task:* Search for a round-trip flight from Tokyo (TYO) to Honolulu (HNL) for one adult, departing...
  - *Judge:* HTTP probe returned 403 Forbidden — site is blocking automated access.

- **hitechglitz.com**  `HTTP 403 reachable=True`  `conf=0.90`  `→ Skipped`
  - *Task:* Access the homepage of hitechglitz.com and identify any main navigation links....
  - *Judge:* HTTP probe returned 403 Forbidden — site is blocking automated access.

- **programmerinterview.com**  `HTTP 520 reachable=False`  `conf=1.00`  `→ Skipped`
  - *Task:* Find the explanation for 'Interface vs. Abstract Class' in Java and summarize the key diff...
  - *Judge:* The HTTP probe result indicates a 520 server error, meaning the website is currently unreachable.

## WEBSITE_DOWN — unreachable  *(pool size: 328)*

- **international.cnrs.fr**  `HTTP None reachable=False`  `conf=0.95`  `→ Skipped`
  - *Task:* Find the main research themes or scientific disciplines listed on the CNRS website....
  - *Judge:* Website could not be reached: HTTPSConnectionPool(host='international.cnrs.fr', port=443): Max retries exceeded with url: / (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1028)')))

- **appft1.uspto.gov**  `HTTP None reachable=False`  `conf=0.95`  `→ Skipped`
  - *Task:* Find patent applications related to 'quantum computing' published after January 1, 2023....
  - *Judge:* Website could not be reached: connection_error

- **garanteprivacy.it**  `HTTP None reachable=False`  `conf=0.95`  `→ Skipped`
  - *Task:* Find the online service for reporting a data breach on the Garante Privacy website and ide...
  - *Judge:* Website could not be reached: connection_error

- **gamedev.jobs**  `HTTP None reachable=False`  `conf=0.95`  `→ Skipped`
  - *Task:* Find at least three active and accessible online job boards or platforms that list Unity g...
  - *Judge:* Website could not be reached: HTTPSConnectionPool(host='gamedev.jobs', port=443): Max retries exceeded with url: / (Caused by SSLError(SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: Hostname mismatch, certificate is not valid for 'gamedev.jobs'. (_ssl.c:1028)")))

## CONTENT_LIKELY_OUTDATED  *(pool size: 338)*

- **dailysignal.com**  `HTTP 200 reachable=True`  `conf=0.90`  `→ Skipped`
  - *Task:* Who wrote the article 'Trump Bypasses Washington Globalists, Takes on China’s Economic Thr...
  - *Judge:* The article 'Trump Bypasses Washington Globalists, Takes on China’s Economic Threat' was not found on the homepage of dailysignal.com. The articles displayed are more recent, suggesting the target article is no longer readily accessible from the main page.

- **brockhole.co.uk**  `HTTP 200 reachable=True`  `conf=0.90`  `→ Skipped`
  - *Task:* Find out if there are any specific accessibility considerations for visitors using wheelch...
  - *Judge:* The website is undergoing a major transition to 'BeWILDerwood' and explicitly states that no activities or bookings are available during this period, making it likely that previous accessibility information for Brockhole on Windermere is no longer current or easily accessible. There is no clear 'Accessibility' link on the homepage.

- **leadership.ucdavis.edu**  `HTTP 200 reachable=True`  `conf=0.90`  `→ Skipped`
  - *Task:* Find the contact email for the 'Leadership Recruitment and Diversity Services Team' mentio...
  - *Judge:* The website is operational, but the specific team 'Leadership Recruitment and Diversity Services Team' and related articles or contact information are not present on the browsed aggiesquare.ucdavis.edu page, suggesting the content is no longer available.

- **cityofalbany.net**  `HTTP 200 reachable=True`  `conf=0.90`  `→ Skipped`
  - *Task:* How can a resident report a street light out in the City of Albany?...
  - *Judge:* The website is operational and the functionality to report a street light out is present, but the navigation steps provided are outdated as a direct link to 'Street light out' is available on the homepage under a 'Report' section, bypassing the need to navigate through 'Your Government' and 'Public Works'.

## UNCERTAIN  *(pool size: 782)*

- **stratford.ca**  `HTTP 200 reachable=True`  `conf=1.00`  `→ Skipped`
  - *Task:* Find the general contact email address for the City of Stratford....
  - *Judge:* The starting page loaded successfully and contains multiple 'Contact Us' links, but the general contact email address is not directly visible on this page, requiring further navigation to confirm its presence.

- **anthempress.com**  `HTTP 200 reachable=True`  `conf=1.00`  `→ Skipped`
  - *Task:* Find the main Anthem Press website's 'About Us' page and identify their core mission or pu...
  - *Judge:* The starting page loaded successfully, but the browsing tool is unable to access the 'About Us' page, preventing further navigation and completion of the task's success criteria.

- **csoonline.com**  `HTTP 200 reachable=True`  `conf=0.90`  `→ Skipped`
  - *Task:* Find a resource on CSO Online that discusses data security in the context of Generative AI...
  - *Judge:* The website is operational and the 'Resources' link is visible, suggesting the initial navigation steps are completable, but further navigation and searching are required to confirm the presence of the specific article.

- **portal.linz.gv.at**  `HTTP 200 reachable=True`  `conf=0.90`  `→ Skipped`
  - *Task:* Find out what documents are required to apply for a 'Meldezettel' (registration certificat...
  - *Judge:* The starting page loaded successfully and mentions 'Service A-Z' in its navigation. However, the browsing tool does not provide link URLs, preventing further navigation to the 'Service A-Z' section to verify the task's completability.

## FEASIBLE — low confidence (rejected)  *(pool size: 202)*

- **ombudsman.hk**  `HTTP 200 reachable=True`  `conf=0.90`  `→ Skipped`
  - *Task:* Find a summary report for a direct investigation operation related to 'Occupational Safety...
  - *Judge:* The website is operational and the specific report 'Government’s Regulation of Occupational Safety and Health in Construction Industry' (政府對建造業職業安全及健康的監管) is listed under 'Direct Investigation Operations' (主動調查行動) with a downloadable summary (摘要). The publication year is likely 2025, as hinted in the success criteria and supported by the website's…

- **autotrader.ca**  `HTTP 200 reachable=True`  `conf=0.90`  `→ Skipped`
  - *Task:* Find the cheapest 2018 Honda Civic for sale within 100 km of Toronto, ON, and report its p...
  - *Judge:* The website is operational and the main search filters for Make, Model, and a postal code input are clearly visible on the homepage. While the year filter and radius setting are not immediately visible, the presence of the primary filters suggests they are likely accessible after initial selections or on the search results page, making the task…

- **meded.ucsf.edu**  `HTTP 200 reachable=True`  `conf=0.90`  `→ Skipped`
  - *Task:* Find the phone number for the 'Admissions & Student Experience (AsSET)' department on the ...
  - *Judge:* The website is operational, and the navigation path to 'About Us' and 'Our People' is present on the homepage, suggesting that a staff directory is still accessible to find the required phone number.

- **goretexprofessional.com**  `HTTP 200 reachable=True`  `conf=0.90`  `→ Skipped`
  - *Task:* Identify which GORE-TEX product technology is specifically designed to prevent blood and b...
  - *Judge:* The website is operational, and the landing page clearly identifies 'GORE-TEX CROSSTECH® product technology' as preventing blood and body fluid penetration. Although specific industry standards are not detailed on this initial page, the general statement about meeting standards and the clear navigation path suggest this information would be…

## FEASIBLE — accepted  *(pool size: 532)*

- **bbbnp.org**  `HTTP 200 reachable=True`  `conf=1.00`  `→ Accepted`
  - *Task:* Download the brochure for the National Partners program....
  - *Judge:* The website is operational and the 'National Partners' section is clearly visible in the navigation. Within that section, there is a prominent 'Download the Brochure' link, indicating the task is still completable. [1]

- **kleenex.com.au**  `HTTP 200 reachable=True`  `conf=1.00`  `→ Accepted`
  - *Task:* What is the current advice for disposing of soft plastic Kleenex packaging in Australia?...
  - *Judge:* The website clearly states that the REDcycle scheme is no longer operational and advises disposing of soft plastics in the regular household bin.

- **gahistoricnewspapers.galileo.usg.edu**  `HTTP 200 reachable=True`  `conf=1.00`  `→ Accepted`
  - *Task:* Find any newspaper articles on the Georgia Historic Newspapers website that mention the 'A...
  - *Judge:* The website is operational and clearly displays a search bar and a large collection of historical newspapers, indicating the task can still be completed as described.

- **dkfz.de**  `HTTP 200 reachable=True`  `conf=1.00`  `→ Accepted`
  - *Task:* List the names of all Translational Centers affiliated with the Deutsches Krebsforschungsz...
  - *Judge:* The website is operational, and the 'Translationale Zentren' page is directly accessible from the main navigation under 'Forschung', listing all six required centers as per the success criteria. [1]

