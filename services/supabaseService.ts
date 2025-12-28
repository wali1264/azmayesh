
import { supabase } from '../utils/supabaseClient';
import * as db from '../utils/db';
import type { 
    Product, ProductBatch, SaleInvoice, PurchaseInvoice, Supplier, Customer, 
    Employee, Expense, Service, Role, User, StoreSettings, ActivityLog, 
    CustomerTransaction, SupplierTransaction, PayrollTransaction, AppState
} from '../types';

export interface AdminProfile {
    id: string;
    email: string;
    is_approved: boolean;
    current_device_id: string | null;
}

// DEFAULT ADMIN ROLE (Local Fallback)
const DEFAULT_ADMIN_ROLE: Role = {
    id: 'admin-role',
    name: 'Admin',
    permissions: [
        'page:dashboard', 'page:inventory', 'page:pos', 'page:purchases', 'page:accounting', 'page:reports', 'page:settings',
        'inventory:add_product', 'inventory:edit_product', 'inventory:delete_product',
        'pos:create_invoice', 'pos:edit_invoice', 'pos:apply_discount', 'pos:create_credit_sale',
        'purchase:create_invoice', 'purchase:edit_invoice',
        'accounting:manage_suppliers', 'accounting:manage_customers', 'accounting:manage_payroll', 'accounting:manage_expenses',
        'settings:manage_store', 'settings:manage_users', 'settings:manage_backup', 'settings:manage_services', 'settings:manage_alerts'
    ]
};

const DEFAULT_SETTINGS: StoreSettings = {
    storeName: 'کتابستان',
    address: '',
    phone: '',
    lowStockThreshold: 10,
    expiryThresholdMonths: 3,
    currencyName: 'افغانی',
    currencySymbol: 'AFN'
};

export const api = {
    // --- ADMIN AUTH & PROFILES (Online Only) ---
    getProfile: async (userId: string): Promise<AdminProfile | null> => {
        try {
            const { data, error } = await supabase.from('profiles').select('*').eq('id', userId).maybeSingle();
            if (error) return null;
            return data;
        } catch (e) {
            return null;
        }
    },
    updateProfile: async (userId: string, updates: Partial<AdminProfile>): Promise<boolean> => {
        try {
            const { error } = await supabase.from('profiles').update(updates).eq('id', userId);
            if (error) {
                console.error("Supabase update error:", error);
                return false;
            }
            return true;
        } catch (e) {
            console.error("Profile update failed:", e);
            return false;
        }
    },
    
    // --- STAFF AUTH (100% LOCAL) ---
    verifyStaffCredentials: async (username: string, password: string): Promise<User | null> => {
        const users = await db.getAll<User>(db.STORES.USERS);
        const user = users.find(u => u.username === username && u.password === password);
        return user || null;
    },

    // --- SETTINGS (Local) ---
    getSettings: async () => {
        const settings = await db.getById<StoreSettings>(db.STORES.SETTINGS, 'current');
        return settings || DEFAULT_SETTINGS;
    },
    updateSettings: async (settings: StoreSettings) => {
        await db.putItem(db.STORES.SETTINGS, { ...settings, id: 'current' });
    },

    // --- USERS & ROLES (LOCAL) ---
    getUsers: async () => {
        return db.getAll<User>(db.STORES.USERS);
    },
    getRoles: async () => {
        const roles = await db.getAll<Role>(db.STORES.ROLES);
        if (roles.length === 0) {
            await db.putItem(db.STORES.ROLES, DEFAULT_ADMIN_ROLE);
            return [DEFAULT_ADMIN_ROLE];
        }
        return roles;
    },
    addUser: async (user: Omit<User, 'id'>) => {
        const newId = crypto.randomUUID();
        const newUser = { ...user, id: newId };
        await db.putItem(db.STORES.USERS, newUser);
        return newUser;
    },
    updateUser: async (user: Partial<User> & { id: string }) => {
        const existing = await db.getById<User>(db.STORES.USERS, user.id);
        if (existing) {
            await db.putItem(db.STORES.USERS, { ...existing, ...user });
        }
    },
    deleteUser: async (id: string) => {
        await db.deleteItem(db.STORES.USERS, id);
    },
    addRole: async (role: Omit<Role, 'id'>) => {
        const newId = crypto.randomUUID();
        const newRole = { ...role, id: newId };
        await db.putItem(db.STORES.ROLES, newRole);
        return newRole;
    },
    updateRole: async (role: Role) => {
        await db.putItem(db.STORES.ROLES, role);
    },
    deleteRole: async (id: string) => {
        if (id === 'admin-role') return;
        await db.deleteItem(db.STORES.ROLES, id);
    },

    // --- SHOP ENTITIES (IndexedDB - Local Storage) ---
    getProducts: async () => db.getAll<Product>(db.STORES.PRODUCTS),
    addProduct: async (product: Omit<Product, 'id'|'batches'>, firstBatch: Omit<ProductBatch, 'id'>) => {
        const productId = crypto.randomUUID();
        const batchId = crypto.randomUUID();
        const newProduct: Product = { ...product, id: productId, batches: [{ ...firstBatch, id: batchId }] };
        await db.putItem(db.STORES.PRODUCTS, newProduct);
        return newProduct;
    },
    updateProduct: async (product: Product) => db.putItem(db.STORES.PRODUCTS, product),
    deleteProduct: async (id: string) => db.deleteItem(db.STORES.PRODUCTS, id),

    getServices: async () => db.getAll<Service>(db.STORES.SERVICES),
    addService: async (service: Omit<Service, 'id'>) => {
        const id = crypto.randomUUID();
        const newService = { ...service, id };
        await db.putItem(db.STORES.SERVICES, newService);
        return newService;
    },
    deleteService: async (id: string) => db.deleteItem(db.STORES.SERVICES, id),

    getEntities: async () => {
        const [customers, suppliers, employees, expenses] = await Promise.all([
            db.getAll<Customer>(db.STORES.CUSTOMERS),
            db.getAll<Supplier>(db.STORES.SUPPLIERS),
            db.getAll<Employee>(db.STORES.EMPLOYEES),
            db.getAll<Expense>(db.STORES.EXPENSES)
        ]);
        return { customers, suppliers, employees, expenses };
    },
    addCustomer: async (c: any) => { const id = crypto.randomUUID(); const item = { ...c, id, balance: 0 }; await db.putItem(db.STORES.CUSTOMERS, item); return item; },
    deleteCustomer: async (id: string) => db.deleteItem(db.STORES.CUSTOMERS, id),
    addSupplier: async (s: any) => { const id = crypto.randomUUID(); const item = { ...s, id, balance: 0 }; await db.putItem(db.STORES.SUPPLIERS, item); return item; },
    deleteSupplier: async (id: string) => db.deleteItem(db.STORES.SUPPLIERS, id),
    addEmployee: async (e: any) => { const id = crypto.randomUUID(); const item = { ...e, id, balance: 0 }; await db.putItem(db.STORES.EMPLOYEES, item); return item; },
    addExpense: async (e: any) => { const id = crypto.randomUUID(); const item = { ...e, id }; await db.putItem(db.STORES.EXPENSES, item); return item; },

    getTransactions: async () => {
        const [customerTransactions, supplierTransactions, payrollTransactions] = await Promise.all([
            db.getAll<CustomerTransaction>(db.STORES.CUSTOMER_TX),
            db.getAll<SupplierTransaction>(db.STORES.SUPPLIER_TX),
            db.getAll<PayrollTransaction>(db.STORES.PAYROLL_TX)
        ]);
        return { customerTransactions, supplierTransactions, payrollTransactions };
    },

    getInvoices: async () => {
        const [saleInvoices, purchaseInvoices] = await Promise.all([
            db.getAll<SaleInvoice>(db.STORES.SALE_INVOICES),
            db.getAll<PurchaseInvoice>(db.STORES.PURCHASE_INVOICES)
        ]);
        return { 
            saleInvoices: saleInvoices.sort((a,b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()), 
            purchaseInvoices: purchaseInvoices.sort((a,b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
        };
    },

    getActivities: async () => {
        const logs = await db.getAll<ActivityLog>(db.STORES.ACTIVITY);
        return logs.sort((a,b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()).slice(0, 100);
    },
    addActivity: async (log: ActivityLog) => db.putItem(db.STORES.ACTIVITY, log),

    createSale: async (invoice: SaleInvoice, stockUpdates: {batchId: string, newStock: number}[], customerUpdate?: {id: string, newBalance: number, transaction: CustomerTransaction}) => {
        await db.putItem(db.STORES.SALE_INVOICES, invoice);
        for (const update of stockUpdates) {
            const product = await findProductByBatchId(update.batchId);
            if (product) {
                product.batches = product.batches.map(b => b.id === update.batchId ? { ...b, stock: update.newStock } : b);
                await db.putItem(db.STORES.PRODUCTS, product);
            }
        }
        if (customerUpdate) {
            const customer = await db.getById<Customer>(db.STORES.CUSTOMERS, customerUpdate.id);
            if (customer) {
                await db.putItem(db.STORES.CUSTOMERS, { ...customer, balance: customerUpdate.newBalance });
                await db.putItem(db.STORES.CUSTOMER_TX, customerUpdate.transaction);
            }
        }
    },

    updateSale: async (invoiceId: string, newInvoiceData: SaleInvoice, stockRestores: {productId: string, quantity: number}[], stockDeductions: {batchId: string, quantity: number}[], customerUpdate?: {id: string, oldAmount: number, newAmount: number, transactionDescription: string}) => {
        await db.putItem(db.STORES.SALE_INVOICES, newInvoiceData);
        for (const restore of stockRestores) {
            const p = await db.getById<Product>(db.STORES.PRODUCTS, restore.productId);
            if (p && p.batches.length > 0) {
                p.batches[0].stock += restore.quantity;
                await db.putItem(db.STORES.PRODUCTS, p);
            }
        }
        if (customerUpdate) {
            const customer = await db.getById<Customer>(db.STORES.CUSTOMERS, customerUpdate.id);
            if (customer) {
                const newBalance = customer.balance - customerUpdate.oldAmount + customerUpdate.newAmount;
                await db.putItem(db.STORES.CUSTOMERS, { ...customer, balance: newBalance });
            }
        }
    },

    updateSaleInvoiceMetadata: async (invoiceId: string, updates: { original_invoice_id?: string | null }) => {
        const inv = await db.getById<SaleInvoice>(db.STORES.SALE_INVOICES, invoiceId);
        if (inv) await db.putItem(db.STORES.SALE_INVOICES, { ...inv, originalInvoiceId: updates.original_invoice_id || undefined });
    },

    createSaleReturn: async (returnInvoice: SaleInvoice, stockRestores: {productId: string, quantity: number}[], customerRefund?: {id: string, amount: number}) => {
        await db.putItem(db.STORES.SALE_INVOICES, returnInvoice);
        for (const restore of stockRestores) {
            const p = await db.getById<Product>(db.STORES.PRODUCTS, restore.productId);
            if (p && p.batches.length > 0) {
                p.batches[0].stock += restore.quantity;
                await db.putItem(db.STORES.PRODUCTS, p);
            }
        }
        if (customerRefund) {
            const customer = await db.getById<Customer>(db.STORES.CUSTOMERS, customerRefund.id);
            if (customer) {
                await db.putItem(db.STORES.CUSTOMERS, { ...customer, balance: customer.balance - customerRefund.amount });
            }
        }
    },

    createPurchase: async (invoice: PurchaseInvoice, supplierUpdate: {id: string, newBalance: number, transaction: SupplierTransaction}, newBatches: any[]) => {
        await db.putItem(db.STORES.PURCHASE_INVOICES, invoice);
        await db.putItem(db.STORES.SUPPLIER_TX, supplierUpdate.transaction);
        const supplier = await db.getById<Supplier>(db.STORES.SUPPLIERS, supplierUpdate.id);
        if (supplier) await db.putItem(db.STORES.SUPPLIERS, { ...supplier, balance: supplierUpdate.newBalance });
        
        for (const b of newBatches) {
            const p = await db.getById<Product>(db.STORES.PRODUCTS, b.productId);
            if (p) {
                p.batches.push(b);
                await db.putItem(db.STORES.PRODUCTS, p);
            }
        }
    },

    updatePurchase: async (invoiceId: string, newInvoiceData: PurchaseInvoice, supplierUpdate?: {id: string, oldAmount: number, newAmount: number}) => {
        await db.putItem(db.STORES.PURCHASE_INVOICES, newInvoiceData);
        if (supplierUpdate) {
            const supplier = await db.getById<Supplier>(db.STORES.SUPPLIERS, supplierUpdate.id);
            if (supplier) await db.putItem(db.STORES.SUPPLIERS, { ...supplier, balance: supplier.balance - supplierUpdate.oldAmount + supplierUpdate.newAmount });
        }
    },

    createPurchaseReturn: async (returnInvoice: PurchaseInvoice, stockDeductions: {productId: string, quantity: number, lotNumber: string}[], supplierRefund?: {id: string, amount: number}) => {
        await db.putItem(db.STORES.PURCHASE_INVOICES, returnInvoice);
        for (const deduct of stockDeductions) {
            const p = await db.getById<Product>(db.STORES.PRODUCTS, deduct.productId);
            if (p) {
                const batch = p.batches.find(b => b.lotNumber === deduct.lotNumber);
                if (batch) batch.stock = Math.max(0, batch.stock - deduct.quantity);
                await db.putItem(db.STORES.PRODUCTS, p);
            }
        }
        if (supplierRefund) {
            const supplier = await db.getById<Supplier>(db.STORES.SUPPLIERS, supplierRefund.id);
            if (supplier) await db.putItem(db.STORES.SUPPLIERS, { ...supplier, balance: supplier.balance - supplierRefund.amount });
        }
    },

    processPayment: async (entityType: 'customer' | 'supplier' | 'employee', entityId: string, newBalance: number, transaction: any) => {
        const store = entityType === 'customer' ? db.STORES.CUSTOMERS : (entityType === 'supplier' ? db.STORES.SUPPLIERS : db.STORES.EMPLOYEES);
        const txStore = entityType === 'customer' ? db.STORES.CUSTOMER_TX : (entityType === 'supplier' ? db.STORES.SUPPLIER_TX : db.STORES.PAYROLL_TX);
        
        const entity = await db.getById<any>(store, entityId);
        if (entity) {
            await db.putItem(store, { ...entity, balance: newBalance });
            await db.putItem(txStore, transaction);
        }
    },

    processPayroll: async (updates: {id: string, balance: 0}[], transactions: PayrollTransaction[], expense: Expense) => {
        for (const u of updates) {
            const emp = await db.getById<Employee>(db.STORES.EMPLOYEES, u.id);
            if (emp) await db.putItem(db.STORES.EMPLOYEES, { ...emp, balance: 0 });
        }
        for (const tx of transactions) await db.putItem(db.STORES.PAYROLL_TX, tx);
        await db.putItem(db.STORES.EXPENSES, expense);
    },

    clearAndRestoreData: async (data: AppState) => {
        await Promise.all(Object.values(db.STORES).map(s => db.clearStore(s)));
        if (data.storeSettings) await db.putItem(db.STORES.SETTINGS, { ...data.storeSettings, id: 'current' });
        for (const p of data.products) await db.putItem(db.STORES.PRODUCTS, p);
        for (const s of data.saleInvoices) await db.putItem(db.STORES.SALE_INVOICES, s);
        for (const p of data.purchaseInvoices) await db.putItem(db.STORES.PURCHASE_INVOICES, p);
        for (const c of data.customers) await db.putItem(db.STORES.CUSTOMERS, c);
        for (const s of data.suppliers) await db.putItem(db.STORES.SUPPLIERS, s);
        for (const e of data.employees) await db.putItem(db.STORES.EMPLOYEES, e);
        for (const e of data.expenses) await db.putItem(db.STORES.EXPENSES, e);
        for (const s of data.services) await db.putItem(db.STORES.SERVICES, s);
        for (const t of data.customerTransactions) await db.putItem(db.STORES.CUSTOMER_TX, t);
        for (const t of data.supplierTransactions) await db.putItem(db.STORES.SUPPLIER_TX, t);
        for (const t of data.payrollTransactions) await db.putItem(db.STORES.PAYROLL_TX, t);
        for (const a of data.activities) await db.putItem(db.STORES.ACTIVITY, a);
    }
};

async function findProductByBatchId(batchId: string): Promise<Product | undefined> {
    const products = await db.getAll<Product>(db.STORES.PRODUCTS);
    return products.find(p => p.batches.some(b => b.id === batchId));
}
