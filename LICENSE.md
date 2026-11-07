# JINX END USER LICENSE AGREEMENT (EULA)

- **Product:** JINX — Enterprise Sovereign Agent Runtime
- **Licensor:** JINX Enterprise Team
- **License version:** 1.0
- **Effective date:** 07.11.2026
- **Contact:** hardincorporated@gmail.com

This Agreement is made in English and in Russian ([LICENSE_RU.md](LICENSE_RU.md)). If the two texts differ, the English text prevails.

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

This Agreement applies only to the versions and components of JINX that the Licensor has expressly designated as distributed under this Agreement (each a "Licensed Version"). Components distributed under separate licenses are governed by their own terms.

In this Agreement, "Licensor" is the rights holder named above; "User" (or "you") is the individual or legal entity that installs, runs or otherwise uses the Program; "Commercial Use" is any use that falls outside the free uses listed in section 2.1, including use in the commercial activity of a company or other organization (section 2.2); "Commercial License" is a license issued or documented by the Licensor under section 2.2; "Output" is the result of the Program's work, including generated source code, text, configurations, files and other materials (section 7).

## 2. Grant of License

Subject to the terms of this Agreement, the Licensor grants the User a limited, non-exclusive, non-transferable license to use the Program.

The license is granted on the following terms.

### 2.1. Free Use

The free license permits:

- personal use of the Program;
- studying its architecture and principles of operation;
- educational use;
- testing and evaluating its functionality;
- developing non-commercial projects;
- use for non-commercial research;
- developing and testing your own extensions for personal use.

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

Use of the Program in the commercial activity of a company is Commercial Use even if running the Program does not itself bring separate revenue.

### 2.3. No Automatic Royalties

This Agreement does not impose any automatic percentage or revenue share on the User's revenue from software created using JINX.

The fee for Commercial Use is set by the terms of the Commercial License that was obtained.

Royalties or other payments apply only where a separate written agreement between the parties expressly provides for them.

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

## 6. Distribution and Integration

JINX may be distributed as part of third-party software only under a license that expressly permits it.

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

The Licensor may terminate the license in the event of a material breach of its terms by the User if the breach has not been cured within 30 calendar days after the User received written notice, where applicable law allows such a procedure.

On termination of the license, the User must:

1. stop using the relevant licensed version of the Program;
2. delete its copies, unless a separate agreement or mandatory law provides otherwise;
3. stop distributing components that the User has no right to distribute.

Termination of the license does not deprive the User of rights in the independent programs, source data and results of work that the User created.

Terms that by their nature must continue after termination of the license remain in force.

## 13. Disclaimer of Warranties

To the maximum extent permitted by applicable law, the Program is provided "AS IS" and "AS AVAILABLE".

The Licensor does not warrant:

- uninterrupted operation of the Program;
- the absence of software errors;
- compatibility with every development environment;
- that an AI agent will achieve any particular result;
- the correctness and completeness of generated code;
- the absence of errors in the actions of external language models;
- that the Program meets the User's particular requirements.

Nothing in this section limits warranties that cannot be excluded under applicable law.

## 14. Limitation of Liability

To the maximum extent permitted by applicable law, the Licensor is not liable for indirect, incidental, special or consequential damages, including:

- loss of data;
- loss of profit;
- downtime of production processes;
- damage to software;
- failures of information systems;
- the consequences of commands executed by an AI agent;
- losses caused by the use of external language models and third-party services.

The Licensor's total liability under this Agreement may not exceed [STATE THE LIABILITY LIMIT OR THE FORMULA FOR CALCULATING IT], within the limits permitted by applicable law.

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

This Agreement is governed by the law of [STATE THE COUNTRY OR JURISDICTION].

Disputes arising in connection with this Agreement are resolved by negotiation and, if no agreement can be reached, in the competent court of [STATE THE JURISDICTION AND PLACE].

Nothing in this Agreement limits the mandatory rights that consumer protection law gives the User.

## 19. Final Provisions

If any provision of this Agreement is held invalid or unenforceable, the remaining provisions remain in force.

A failure by the Licensor to exercise a right does not waive that right.

This Agreement contains the basic licensing terms of the relevant version of JINX and is supplemented by separate commercial agreements where the parties have concluded them.

---

## 20. Contact Information

**Software product:** JINX — Enterprise Sovereign Agent Runtime

**Email:** hardincorporated@gmail.com

**Official repository:** https://github.com/machinegpt/agent

© 2026 JINX Enterprise Team. All rights reserved, subject to applicable third-party licenses.
