# JINX END USER LICENSE AGREEMENT (EULA)

- **Product:** JINX — Enterprise Sovereign Agent Runtime
- **Licensor:** JINX Enterprise Team
- **License version:** 1.0
- **Effective date:** 7 November 2026
- **Contact:** hardincorporated@gmail.com

This Agreement is made in English, in Russian ([LICENSE_RU.md](LICENSE_RU.md)) and in Chinese ([LICENSE_ZH.md](LICENSE_ZH.md)). If the texts differ, the English text prevails.

**Please read sections 4 (Restrictions), 13 (Disclaimer of Warranties), 14 (Limitation of Liability) and 18 (Governing Law and Dispute Resolution) with particular attention.**

---

## 1. General Provisions

This End User License Agreement (the "Agreement") governs the use of the JINX software (the "Program").

The Program is a runtime environment for AI agents (an Agent Runtime) designed for integration with IDEs, command-line tools (CLI), code editors, enterprise systems and other software environments.

The Program includes:

- the Runtime Engine and the mechanisms that execute the agent loop;
- the system for managing state and preserving context;
- the inter-process communication (IPC) mechanisms;
- the protocols for exchanging data with host applications;
- the systems for managing tools and function calls;
- the interfaces for integration with language models;
- the source code, executable components and documentation distributed by the Licensor.

By installing, copying, running or otherwise using the Program, the User confirms that the User agrees to this Agreement.

A User who does not agree with these terms must stop using the relevant licensed version of the Program.

This Agreement is an offer addressed to an unlimited number of persons. The User accepts it, and the Agreement is concluded, by installing, copying, running or otherwise using the Program, or by any other conduct that shows agreement. No signature is needed. Where the law requires a written form, this conduct meets that requirement to the extent the law allows.

This Agreement applies only to the versions and components of JINX that the Licensor has expressly designated as distributed under this Agreement (each a "Licensed Version"). Components distributed under separate licenses are governed by their own terms.

JINX version 1.3.0 and every later version that is published with this Agreement are Licensed Versions. The Agreement is published as the file LICENSE.md in the repository https://github.com/machinegpt/agent and, in a project where JINX has been installed, in the `.agent` folder. Versions up to and including 1.2.9 were published under the Apache License, Version 2.0. This Agreement does not change the license of those versions, and a copy that a User has already received under that license stays under it.

In this Agreement, "Licensor" is the holder of the rights in JINX, who publishes it under the name "JINX Enterprise Team" (the Licensor gives its legal name and address to a User in writing, on a request sent to the contact address in section 20, where this is needed to conclude a Commercial License or to enforce this Agreement); "User" (or "you") is the individual or legal entity that installs, runs or otherwise uses the Program; "Commercial Use" is any use that falls outside the free uses listed in section 2.1, including use in the commercial activity of a company or other organization (section 2.2); "Commercial License" is a license issued or documented by the Licensor under section 2.2; "Output" is the result of the Program's work, including generated source code, text, configurations, files and other materials (section 7).

## 2. Grant of License

Subject to the terms of this Agreement, the Licensor grants the User a limited, non-exclusive, non-transferable license to use the Program.

The license is granted on the following terms.

### 2.1. Free Use

The free license permits:

- personal use of the Program;
- studying its architecture and principles of operation;
- educational use;
- testing and evaluating its functionality, including evaluation by a company or other organization for up to 30 days in a non-production environment, before it decides whether to buy a Commercial License;
- developing non-commercial projects;
- use for non-commercial research;
- developing and testing your own extensions for personal use.

A use is non-commercial if it is not directed toward commercial advantage or monetary compensation and is not part of the work of a business.

The free license does not automatically permit the commercial distribution of the Program itself or its inclusion in commercial products.

### 2.2. Commercial Use

Commercial Use is permitted if you hold a valid Commercial License issued or documented by the Licensor.

A Commercial License may take the form of:

- an individual developer license;
- a development team license;
- a corporate license;
- a license for integration into third-party software;
- a license for the distribution of embedded components;
- a special OEM and White Label license.

The specific terms, price, number of users, term and permitted scale of use are set by a separate order, contract or commercial offer.

Commercial Use includes, in particular, use of the Program:

- in the business of a company, an individual entrepreneur or another organization that carries on commercial activity, by that organization or by its employees and contractors;
- to develop, maintain or deliver software or services for clients for a fee or other consideration;
- to offer the Program or its functions to third parties for a fee, including as a hosted or managed service;
- as part of a product or service that is sold, licensed or otherwise offered for a fee (see also section 6).

Use of the Program in the commercial activity of a company is Commercial Use even if running the Program does not itself bring separate revenue, except for the evaluation permitted by section 2.1.

To obtain a Commercial License, write to the contact address in section 20. The Licensor decides whether to offer a Commercial License and on what terms. A Commercial License takes effect when the Licensor confirms it in writing (an e-mail is enough) or issues an order and the User accepts it.

Commercial Use without a Commercial License is not covered by the license granted in this Agreement. The User must stop that use or obtain a Commercial License. The Licensor may claim payment for the period of that use at the price the Licensor then charges for the license that fits it, in addition to the Licensor's other remedies.

### 2.3. No Automatic Royalties

This Agreement does not impose any automatic percentage or revenue share on the User's revenue from software created using JINX.

The fee for Commercial Use is set by the terms of the Commercial License that was obtained.

Royalties or other payments apply only where a separate written agreement between the parties expressly provides for them.

### 2.4. Territory, Term and Sublicensing

The license is granted for the whole world.

The license is granted for the whole term of the Licensor's exclusive rights in the Program, unless it is terminated earlier under section 12. A Commercial License lasts for the term stated in it.

The license for the free uses in section 2.1 is granted free of charge.

The User may not sublicense the rights granted by this Agreement unless a Commercial License expressly allows it. A person who receives the Program from the User receives the license directly from the Licensor, on the terms of this Agreement, and must accept this Agreement for themselves.

## 3. Permitted Use

Under an applicable license, the User may:

1. install and run the Program on compatible devices;
2. use JINX as part of the User's own software projects;
3. integrate the Runtime into IDEs, CLI tools, code editors and other applications;
4. connect the User's own and third-party language models through the supported integration mechanisms;
5. use the IPC protocols to interact with external host applications;
6. create the User's own tools, extensions and integrations;
7. modify the software components for internal use within the limits of the User's license;
8. create the User's own applications using the JINX APIs and interfaces;
9. use source code created with the help of JINX in personal and commercial projects, subject to all other applicable licenses.

## 4. Restrictions

Without the Licensor's prior written permission, the User may not:

1. sell, rent out or give third parties standalone access to a licensed version of JINX;
2. distribute the source code of the Program, unless the User's license grants that right;
3. publish modified versions of JINX as standalone products;
4. create derivative commercial products whose main function is to resell or provide JINX itself;
5. remove, alter or obscure copyright notices and notices of ownership of the Program;
6. use the JINX name, logos and trademarks in a way that implies official support or endorsement by the Licensor;
7. circumvent technical licensing restrictions, authorization mechanisms or limits on the number of permitted users;
8. use the Program to violate applicable law or the rights of third parties.

The restrictions of this section do not limit any right of the User that cannot be limited under mandatory provisions of applicable law.

## 5. Source Code and Modifications

The Licensor retains all exclusive rights in the original source code, the architecture, the internal mechanisms and the proprietary software components of JINX.

Granting a license does not transfer ownership of the Program to the User.

The User keeps the rights in the User's own independent developments that were created without copying protected elements of JINX.

For modifications that contain original JINX components, the Licensor's rights and the terms of the applicable license continue to apply.

The User may modify the Program for internal use within the license rights granted to the User.

Any permission for public distribution of modified versions must be expressly provided for by the applicable license or by a separate written agreement.

All rights that this Agreement does not expressly grant are reserved by the Licensor.

**Contributions and feedback.** If the User sends the Licensor a contribution to JINX (for example a pull request, a patch or a suggestion), the User confirms that the User has the right to send it, and grants the Licensor a worldwide, royalty-free, non-exclusive license, for the whole term of the exclusive right and, as far as the law allows, irrevocable, to use, copy, modify, publish, distribute and sublicense the contribution, including as part of JINX under this Agreement or under any other terms the Licensor chooses, with the right to transfer this license. The User keeps the User's own rights in the contribution. Unless the User and the Licensor agree otherwise in writing, the Licensor owes no payment for a contribution and does not have to use it.

## 6. Distribution and Integration

JINX may be distributed as part of third-party software only under a license that expressly permits it.

**Installed copies.** `jinx install` puts a copy of the Program's runtime files, with the text of this Agreement, into the `.agent` folder of the User's project. The User may keep that copy in the project and in its version control, and may share the project with the people who work on it. The User may also publish the project, if the copy in it is unchanged (or changed only by JINX itself). In every case, the notices and the text of this Agreement must stay with the copy, and the project must not be a means to resell or provide JINX itself (section 4). Each person who receives the copy receives the license directly from the Licensor, on the terms of this Agreement, and needs a Commercial License for their own Commercial Use.

A commercial integration license may permit:

- including executable JINX components in an application;
- shipping JINX together with the User's own software;
- internal corporate deployment;
- distributing binary components to end users;
- using JINX as part of a commercial platform.

Unless a separate license expressly provides for distribution of source code, the User has no right to distribute it.

When distributing JINX components, the User must keep the copyright notices required by the license and comply with the licenses of third-party components.

## 7. Rights in Output

All rights in the source data, files, projects and materials that the User supplies to the Program remain with the User or the relevant rights holders.

Output, including generated source code, text, configurations, files and other materials, does not become the property of the Licensor merely because it was created using JINX.

To the maximum extent permitted by applicable law, the Licensor makes no claim to the User's exclusive rights in results that the User created independently.

At the same time:

- the User is solely responsible for the lawfulness of the use of Output;
- the User must review Output before using it;
- the User is solely responsible for complying with the licenses of third-party libraries and components;
- Output produced by external language models may be subject to the additional terms of the providers of those models;
- the Licensor does not guarantee that Output is unique.

## 8. Interaction with External Systems and Security

JINX is designed to work inside a host environment and may interact with it through IPC, file protocols and other provided interfaces.

Depending on the integration, the host application may give JINX access to:

- the file system;
- the execution of operating system commands;
- developer tools;
- external language models;
- databases and other connected services.

The User understands and accepts that such operations may change files, run commands and affect the working environment.

The User is solely responsible for:

1. choosing the permissions granted to the agent;
2. configuring access rights;
3. protecting API keys and credentials;
4. reviewing commands and changes before they are applied;
5. keeping backups of important data;
6. setting up an isolated execution environment where one is needed.

The Licensor does not guarantee the absolute safety of decisions, actions and results obtained using AI agents.

## 9. Third-Party Components

The Program may use third-party libraries, tools, programming interfaces and other components.

Such components may be distributed under separate licenses.

This Agreement does not cancel, replace or change the terms of third-party licenses.

If this Agreement conflicts with the mandatory terms of a separate license of a third-party component, the terms of that third-party license apply to that component.

## 10. Updates and Technical Support

The Licensor may release updates, fixes, improvements and new versions of JINX.

The Licensor has no obligation to provide updates or technical support unless a separate contract provides otherwise.

Updates may change functionality, architecture, interaction protocols and system requirements.

The Licensor may end support for individual versions of the Program.

## 11. Fees and Payments

The free uses provided for in section 2.1 require no payment.

Commercial Use is governed by the terms of the Commercial License that was obtained.

The price of a license, the method of payment, the term, the renewal procedure and other financial terms are set by a separate commercial offer or contract.

Unless the parties agree otherwise, taxes, duties and other mandatory payments are paid in accordance with applicable law.

## 12. Termination

This Agreement remains in force until it is terminated.

The Licensor may terminate the license in the event of a material breach of its terms by the User if the breach has not been cured within 30 calendar days after the User received written notice, where applicable law allows such a procedure. If a breach cannot be cured (for example, the unauthorized distribution of the source code of the Program), the Licensor may terminate the license at once by written notice, where applicable law allows.

The User may end this Agreement at any time by deleting all copies of the Program.

On termination of the license, the User must:

1. stop using the relevant licensed version of the Program;
2. delete its copies, unless a separate agreement or mandatory law provides otherwise;
3. stop distributing components that the User has no right to distribute.

Termination of the license does not deprive the User of rights in the independent programs, source data and results of work that the User created.

Terms that by their nature must continue after termination of the license (in particular sections 5, 7, 13, 14, 18 and 19) remain in force.

## 13. Disclaimer of Warranties

To the maximum extent permitted by applicable law, the Program is provided "AS IS" and "AS AVAILABLE".

The Licensor does not warrant:

- uninterrupted operation of the Program;
- the absence of software errors;
- compatibility with every development environment;
- that an AI agent will achieve any particular result;
- the correctness and completeness of generated code;
- the absence of errors in the actions of external language models;
- that the Program meets the User's particular requirements;
- that the Program does not infringe the rights of third parties.

Nothing in this section limits warranties that cannot be excluded under applicable law, or warranties that a Commercial License expressly gives.

## 14. Limitation of Liability

To the maximum extent permitted by applicable law, the Licensor is not liable for indirect, incidental, special or consequential damages, including:

- loss of data;
- loss of profit;
- downtime of production processes;
- damage to software;
- failures of information systems;
- the consequences of commands executed by an AI agent;
- losses caused by the use of external language models and third-party services.

The Licensor's total liability under this Agreement, for all claims taken together and whatever their legal basis, may not exceed the greater of: (a) the total amount that the User actually paid to the Licensor for the Licensed Version concerned in the 12 months before the event that gave rise to the claim; and (b) 100 US dollars (or the equivalent in another currency at the official exchange rate of the Central Bank of the Russian Federation on the date the claim is made). A User who has paid nothing may therefore claim no more than the amount in (b). This applies within the limits permitted by applicable law.

This section does not exclude liability where the law prohibits such an exclusion, including cases of intentional breach, fraud, or liability that cannot be limited by contract.

## 15. Compliance with Law

The User undertakes to use the Program in accordance with applicable law.

The User is solely responsible for:

- complying with data protection law;
- the lawfulness of processing personal information;
- holding the necessary rights in the materials used;
- complying with the requirements that apply to the use of AI systems;
- complying with the export and sanctions restrictions that apply to the User.

## 16. Trademarks and Name

The JINX name, logos, brand marks and other trademarks belonging to the Licensor are not transferred to the User as property.

This Agreement does not grant the right to register or use identical or similar marks as the User's own trademarks.

The JINX name may be used to state, in the ordinary way, that something is compatible with JINX or uses the technology, provided that this does not create a false impression of official approval of a product.

## 17. Changes to the License Terms

The Licensor may publish new editions of this Agreement for later versions of the Program.

A new edition does not retroactively change the terms of a license already granted, unless the parties agree otherwise or applicable law requires it.

Continued use of a new version of the Program for which a new edition has been published means acceptance of the updated terms, where applicable law permits such acceptance.

## 18. Governing Law and Dispute Resolution

This Agreement, and any dispute or claim that arises out of or in connection with it, including non-contractual claims, are governed by the law of the Russian Federation, without regard to its conflict-of-law rules.

Before going to court, a party sends the other a written claim that describes the dispute. The party that receives the claim answers within 30 calendar days. If the dispute is not settled in that period, either party may go to court.

Disputes that are not settled are decided by the Arbitrazh Court of the City of Moscow, where the dispute is within its jurisdiction, and otherwise by the competent court of general jurisdiction in the city of Moscow, Russian Federation.

The Licensor may also bring a claim to protect its intellectual property rights, or ask for an injunction or other interim relief, in any competent court, including the courts of the place where the User is located or where the infringement takes place.

Nothing in this Agreement limits the mandatory rights that consumer protection law gives the User, including the right of a User who is a consumer to go to court without a prior claim and in the court of the User's place of residence.

## 19. Final Provisions

If any provision of this Agreement is held invalid or unenforceable, the remaining provisions remain in force.

A failure by the Licensor to exercise a right does not waive that right.

This Agreement contains the basic licensing terms of the relevant version of JINX and is supplemented by separate commercial agreements where the parties have concluded them. Where a Commercial License differs from this Agreement, the Commercial License prevails for the use it covers.

**Notices.** The User sends notices to the Licensor at the contact address in section 20. The Licensor may send a notice to a User at an e-mail address that the User has given to the Licensor, and may publish a general notice in the repository https://github.com/machinegpt/agent. An e-mail notice is considered received on the day after it is sent, unless the sender gets a message that it was not delivered.

**Assignment.** The User may not assign this Agreement or any right under it without the Licensor's written consent. The Licensor may assign this Agreement and its rights in JINX to another person, including in a merger, sale or reorganization; the User's rights under this Agreement stay in force after that.

---

## 20. Contact Information

**Software product:** JINX — Enterprise Sovereign Agent Runtime

**Email:** hardincorporated@gmail.com (requests for a Commercial License, notices under this Agreement and questions about it)

**Official repository:** https://github.com/machinegpt/agent

© 2026 JINX Enterprise Team. All rights reserved, subject to applicable third-party licenses.
